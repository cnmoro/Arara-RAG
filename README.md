# arara-rag

**A Portuguese-first retrieval stack that runs entirely on CPU.** Chunking,
dense retrieval, lexical retrieval, rank fusion and reranking — numpy only, no
PyTorch, no ONNX Runtime, no FAISS.

```bash
# Not on PyPI yet -- install from a checkout:
pip install -e ".[bench]"
```

```python
from arara_rag import Arara

arara = Arara()                                  # PT-BR defaults
arara.add_documents({"lei": open("lei.txt").read()})
hits = arara.search("qual a alíquota do imposto de renda?", top_k=5)

for h in hits[:3]:
    print(h.score, h.doc_id, arara.resolve(h))   # exact source span
```

`arara.resolve(hit)` returns the exact substring the hit points at, because
every chunk carries character offsets into the canonical source document — not
just the chunk text.

```bash
python -m arara_rag search "licenciamento ambiental" docs/ --json -k 3
```

---

## Why this exists

Brazilian Portuguese is mostly served by English-first tooling and GPU-scale
models. arara is the opposite bet: **retrieval quality per CPU-cycle**. Every
layer is a small model, most of them built specifically for PT-BR, and the whole
thing installs in **200 MB**.

| Stage | Component | Size | Origin |
|---|---|---|---|
| Chunking | [`tinyzchunk`](https://github.com/cnmoro/tinyzchunk) — tokenizer-free, distilled from an LLM teacher | 2.1 MB | cnmoro |
| Dense | [`static-nomic-384-pten-v2`](https://huggingface.co/cnmoro/static-nomic-384-pten-v2) — Model2Vec static embeddings | 62 MB | cnmoro |
| Lexical | BM25 over a numpy inverted index | — | this repo |
| Rerank | [`CXM25`](https://github.com/cnmoro/CXM25) — BM25-inspired lexical scoring with a PT-BR stemmer | bundled | cnmoro |
| Fusion | Reciprocal Rank Fusion | — | this repo |

Also included, both written here against the same index: an exact numpy vector
index and a BM25 inverted index, plus a `rerank()` / `score_documents()` API for
scoring a fixed candidate set.

Nothing here needs a GPU, a tokenizer model server, or a transformer runtime.
A static embedding model is a lookup table, so encoding is tokenize-then-lookup
and search is one matrix multiply.

### Runtime footprint, measured

| | Size |
|---|---|
| arara-rag, clean venv, 26 packages | **200 MB** |
| PyTorch alone | 1.6 GB |
| `torch` + `transformers` + `faiss` (typical RAG stack) | ~4 GB |

---

## Results

All numbers are on **MTEB-BR** (the Brazilian Portuguese embedding benchmark,
[public leaderboard](https://huggingface.co/spaces/MTEB-BR/leaderboard) with 98
evaluated models). Metrics are computed by `bench/metrics.py`, and
`bench/validate_metrics.py` asserts they match `pytrec_eval` to 0.0e+00 — the
same library MTEB uses, including its less obvious choice of **linear** NDCG
gain.

<!-- RESULTS_START -->
### Cross-task, nDCG@10

Fixed-window chunking. These corpora are mostly single-chunk documents, so this
table isolates the retrievers rather than the chunker.

| Task | docs | dense | lexical | hybrid | best |
|---|---|---|---|---|---|
| BRTaxQAR (capped) | 478 | 0.2933 | 0.4051 | 0.3487 | **lexical** |
| FaQuADIR | 244 | 0.7135 | 0.8961 | 0.8304 | **lexical** |
| FaqBacenRetrieval | 1,673 | 0.3745 | 0.4881 | 0.4526 | **lexical** |
| JurisTCU | 16,045 | 0.3887 | 0.5378 | 0.4878 | **lexical** |
| Quati | 50,000 | 0.3268 | 0.4067 | 0.4046 | **lexical** |

Query latency: 0.3–6.7 ms per query on one CPU core, whole corpus scored per
query. Index build: 1.7 s for 244 documents, 139 s for 50,000.

### BRTaxQAR ablation, nDCG@10

Each row changes exactly one thing. "One vector per document" is the operating
point every fixed-window embedder is stuck at.

| Configuration | chunks | nDCG@10 | R@100 |
|---|---|---|---|
| capped at 32k, one vector per doc *(the leaderboard's setting)* | 478 | 0.1496 | 0.4351 |
| capped at 32k, fixed 2500-char windows | 2,552 | 0.2933 | 0.6300 |
| capped at 32k, **tinyzchunk** boundaries | 23,319 | 0.3088 | 0.6225 |
| capped at 32k, tinyzchunk + BM25 | 23,319 | 0.3420 | 0.7446 |
| **full documents**, one vector per doc | 478 | 0.1496 | 0.4351 |
| **full documents**, fixed windows | 6,439 | 0.4040 | 0.7497 |
| **full documents**, paragraph splits | 6,439 | 0.4040 | 0.7497 |
| **full documents**, tinyzchunk | 60,927 | 0.4287 | 0.7486 |
| **full documents**, tinyzchunk + BM25 | 60,927 | 0.4824 | 0.8496 |
| **full documents**, + CXM25 rerank | 60,927 | **0.5091** | 0.8102 |

What this says:

- **One vector per document scores 0.1496 — and the number is identical for
  capped and full input.** Averaging a 300 KB statute into 384 dimensions
  destroys it either way, so truncation is not the only problem. This is the
  worst of the ten configurations, and it is the one the leaderboard is forced
  into.
- **Chunking roughly doubles it**: 0.1496 → 0.2933 (fixed windows) or 0.3088
  (tinyzchunk).
- **Refusing to truncate then adds more**: with the *same* fixed windows,
  0.2933 → 0.4040. The learned chunker adds a further +0.025 on top.
- **The full stack is 3.4× the leaderboard's operating point** on this task.
- **Paragraph splitting ties fixed windows exactly** (0.4040, same chunk count).
  On statutes, paragraphs are far longer than the window, so a "semantic" split
  degenerates into the baseline it was meant to beat. This is the case for a
  learned chunker rather than a formatting heuristic.

### CXM25 reranking earns its place

| Task | hybrid | + CXM25 rerank | Δ |
|---|---|---|---|
| FaQuADIR | 0.8304 | **0.9078** | +0.077 |
| BRTaxQAR (full docs) | 0.4824 | **0.5091** | +0.027 |
| FaqBacenRetrieval | 0.4526 | **0.4707** | +0.018 |
| Quati | 0.4046 | **0.4211** | +0.017 |

CXM25 improves every task it was tried on, for 1–6 ms per query.

### Reranking tasks: MAP@1000

MTEB-BR's reranking tasks hand you a fixed candidate list per query (BM25 hard
negatives) and score only the resulting order, so `identity` is the baseline the
benchmark ships with:

| Task | identity (given order) | dense | lexical | hybrid | **CXM25** |
|---|---|---|---|---|---|
| QuatiReranking | 0.2839 | 0.2798 | 0.2939 | 0.3066 | **0.3100** |
| JurisTCUReranking | 0.4150 | 0.3609 | 0.4279 | 0.4129 | **0.4845** |

CXM25 is the best of arara's rerankers on both tasks, and the only one that beats
the given order on both. Two things worth stating plainly:

- **Dense reranking actively hurts** on JurisTCU (0.4150 → 0.3609). Reranking
  discards the lexical safety net that RRF provides during retrieval, so a weak
  semantic scorer does more damage here than in the retrieval tables above.
- **Against purpose-built rerankers arara is not competitive.** The best
  cross-encoder on the public board (`voyage/rerank-2.5`) scores 0.7560 and
  0.6516 on these two tasks; CXM25 lands at the 11th and 38th percentile. A
  lexical reranker is a cheap improvement over a BM25 order, not a replacement
  for a cross-encoder.

### A negative result: the fusion weights

Equal-weight RRF loses to lexical-only everywhere, so the weights were swept
(dense : lexical, fixed at 1 : w):

| Task | 1:1 | 1:2 | 1:3 | 1:5 | lexical only |
|---|---|---|---|---|---|
| FaQuADIR | 0.8304 | 0.8635 | 0.8762 | 0.8826 | **0.8961** |
| FaqBacen | 0.4526 | 0.4765 | 0.4798 | 0.4845 | **0.4881** |
| BRTaxQAR (capped) | 0.3487 | 0.3679 | 0.3744 | 0.3871 | **0.4051** |
| JurisTCU | 0.4878 | 0.5069 | 0.5216 | 0.5296 | **0.5378** |

The hybrid improves **monotonically as the dense weight goes to zero, and never
overtakes lexical alone**. That is the signature of a retriever contributing
noise rather than signal: on these five tasks the static dense model is not
earning its place.

Two honest caveats before you act on it. The sweep was run on the same tasks
reported above, so the tuned numbers are optimistic and the untuned 1:1 column
is the one to trust in the cross-task table. And these five benchmarks are
short-document, high-lexical-overlap PT-BR tasks — exactly the regime where
BM25 is strongest and a 384-dimension static model is weakest. Dense retrieval
should still help on paraphrase-heavy or cross-lingual queries, which none of
these tasks measure. The default is therefore left untuned at 1:1, and this is
flagged as the first thing worth investigating on your own data.
<!-- RESULTS_END -->

### Where this stands against the leaderboard

MTEB-BR publishes per-task results for 98 models, including commercial APIs.
Placing arara's best configuration among them:

| Task | arara | best on the leaderboard | leaderboard median | models beaten |
|---|---|---|---|---|
| **FaQuADIR** | **0.9078** | voyage-context-4 (0.8738) | 0.7689 | **96 / 96** |
| BRTaxQAR (capped) | 0.4051 | voyage-finance-2 (0.4499) | 0.2723 | 90 / 95 |
| JurisTCU | 0.5378 | llama-embed-nemotron-8b (0.6805) | 0.5436 | 45 / 96 |
| FaqBacenRetrieval | 0.4881 | codestral-embed (0.8262) | 0.6516 | 29 / 96 |
| Quati | 0.4211 | voyage-context-4 (0.6901) | 0.5569 | 28 / 96 |

**On FaQuADIR, arara outranks every model on the board** — above `voyage-context-4`,
`gemini-embedding-2`, `Qwen3-Embedding-8B` and every other entry — at 200 MB and
5.3 ms per query on one CPU core.

Now the caveat, which matters more than the headline: **the leaderboard contains
embedding models only. There is no BM25 entry on it.** arara's strongest modes
are lexical, and lexical retrieval is simply very good on short,
high-lexical-overlap PT-BR tasks. Part of what looks like a win is a missing
baseline on their side rather than a transformer-killing dense model on ours.
arara's *dense* mode alone lands between the 19th and 56th percentile, and that
is the honest measure of the static encoder.

BRTaxQAR is the one row where the comparison is exact: the leaderboard truncates
documents at 32k, `brtaxqa_capped` does the same, and arara's lexical mode
(0.4051) beats both `bge-m3` (0.3772) and `multilingual-e5-large`. The
stack's full-document configuration reaches **0.5091** on input the leaderboard
cannot represent at all.

### Three things to take away

1. **Chunking and truncation matter more than the retriever on long documents.**
   The ten-row BRTaxQAR ablation spans 0.1496 → 0.5091; switching dense for
   lexical moves a single task by at most 0.11.
2. **The learned chunker earns its keep where formatting heuristics fail.**
   Paragraph splitting tied fixed windows exactly; tinyzchunk did not.
3. **CXM25 is the component that pays for itself**, improving every retrieval
   task and beating the given BM25 order on both reranking tasks.
4. **The static dense model does not earn its place on these tasks**, and the
   fusion sweep plus the dense-reranking regression are the evidence. A better
   PT-BR dense model is the single highest-leverage change to this stack.

---

## Guarantees

The chunker's contract is enforced by tests, not asserted in prose
(`tests/`):

- every chunk is an **exact substring** of the canonical document;
- chunks are ordered and non-overlapping;
- the gap between consecutive chunks contains **only whitespace** — no
  non-whitespace character is ever dropped or duplicated;
- **no chunk exceeds `max_chunk_chars`**, including on a 24,000-character
  single line;
- CRLF and LF inputs chunk **identically**, and offsets still resolve.

Three of those required fixing behaviour inherited from upstream:

| Problem | Fix |
|---|---|
| The upstream chunker strips whitespace between structural units, so chunks are not byte-adjacent | Contract redefined as "gaps are whitespace-only" and tested |
| It returns **oversized chunks** on degenerate input (a very long single line) | arara splits oversized pieces at word boundaries before returning |
| It leaves CRLF in the returned text, so `text[start:end] == chunk` fails | Line endings are canonicalised; `Arara.document_text()` exposes what offsets index into |

There is also a regression test for a real bug found while building this: the
BM25 inverted index derived its offsets from raw term occurrences instead of
per-document unique terms, which left **uninitialised memory** in the postings
arrays and corrupted scores. `test_bm25_matches_naive_definition` now compares
the vectorised path against a literal implementation of the BM25 formula.

---

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -e ".[bench]"
python -m pytest tests/                     # 69 tests
python bench/validate_metrics.py            # needs pytrec_eval-terrier
HF_HOME=$PWD/.cache/hf ./bench/run_all.sh   # retrieval suites -> bench/results/
python -m bench.rerank                      # reranking suites (MAP@1000)
python -m bench.report --readme             # the tables above
python -m bench.leaderboard                 # comparison against MTEB-BR
```

`bench/validate_metrics.py` is the only script needing `pytrec_eval`; it is the
check that makes the metric numbers trustworthy, and it is worth running before
believing any score in this README.

The benchmark harness is the only part that needs the heavy stack
(`datasets`, `pyarrow`). The library itself never imports torch or onnxruntime —
`test_importing_the_package_does_not_import_torch` runs in a subprocess and
fails if either appears in `sys.modules`.

---

## Layout

```
arara_rag/
  chunk.py      chunking + the losslessness contract
  dense.py      static encoder + exact numpy index
  lexical.py    BM25 inverted index + CXM25 reranker
  fuse.py       reciprocal rank fusion
  pipeline.py   Arara: add_documents / search / resolve
  cli.py        python -m arara_rag search
bench/
  tasks.py      MTEB-BR task loaders (pinned revisions)
  metrics.py    nDCG / recall / MRR / MAP matching pytrec_eval
  run.py        retrieval experiment suites
  rerank.py     reranking experiment suite (MAP@1000)
  leaderboard.py  comparison against the public leaderboard
  report.py     markdown tables
tests/          contract and correctness tests
space/          Gradio demo for the HuggingFace Space
```

---

## Limitations

- **PT-BR and English only.** The tokenizer, stemmer and stopwords are
  Portuguese; the dense model is EN+PT. Other languages will degrade.
- **The dense model is small.** It will lose to transformer encoders on
  semantic and paraphrase-heavy queries. Lexical retrieval carries this stack.
- **The index is in memory.** It is float16 and fast, but there is no
  disk-backed serving path yet.
- **CXM25 reranking is ~71 µs/document**, so it is applied to a candidate set
  rather than the whole corpus.
- **The BRTaxQAR `document` runs use one vector per document**, which averages
  away most of a 300 KB statute. They are reported to isolate truncation, not as
  a recommended configuration.

## License

Apache-2.0.
