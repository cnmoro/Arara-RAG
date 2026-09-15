"""Behavioural tests for arara-rag.

These assert the guarantees the README claims, not just that the code runs.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from arara_rag import Arara, BM25Index, Chunker, reciprocal_rank_fusion
from arara_rag.chunk import canonicalize


# --------------------------------------------------------------------------
# chunking guarantees
# --------------------------------------------------------------------------
DOCS = {
    "prose": (
        "A Constituição Federal de 1988 estabelece os direitos fundamentais.\n\n"
        "Art. 5º Todos são iguais perante a lei, sem distinção de qualquer natureza.\n\n"
        "Parágrafo único. O disposto neste artigo aplica-se aos estrangeiros.\n\n"
        "Art. 6º São direitos sociais a educação, a saúde, o trabalho e a moradia."
    ),
    "qa": "\n".join(
        f"Pergunta: qual é o prazo do item {i}?\nResposta: o prazo é de {i} dias úteis."
        for i in range(1, 20)
    ),
    "code": (
        "Veja o exemplo:\n\n```python\ndef f(x):\n    return x + 1\n```\n\n"
        "O código acima incrementa o valor recebido."
    ),
    "table": "| col a | col b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n",
    "crlf": "Linha um.\r\nLinha dois.\r\n\r\nLinha três.",
    "empty": "",
    "whitespace": "   \n\n   \t  \n",
    "one_line": "palavra " * 3000,
    "no_spaces": "x" * 5000,
}


def assert_lossless(text: str, chunks) -> None:
    """The documented contract, checked directly."""
    if not text.strip():
        # Nothing but whitespace: there is no content to index.
        assert chunks == []
        return
    assert chunks, "non-empty document produced no chunks"
    for c in chunks:
        assert text[c.start : c.end] == c.text, f"{c.chunk_id} offsets do not resolve"
    for a, b in zip(chunks, chunks[1:]):
        assert a.end <= b.start, "chunks overlap or are out of order"
        gap = text[a.end : b.start]
        assert gap.strip() == "", f"non-whitespace dropped between {a.chunk_id} and {b.chunk_id}"
    # Leading/trailing whitespace may be trimmed, but nothing else.
    assert text[: chunks[0].start].strip() == ""
    assert text[chunks[-1].end :].strip() == ""
    # Reassembling chunks plus the whitespace-only gaps recovers the document,
    # so no non-whitespace character was dropped or duplicated.
    rebuilt = ""
    prev = 0
    for c in chunks:
        rebuilt += text[prev : c.start] + c.text
        prev = c.end
    rebuilt += text[prev:]
    assert rebuilt == text


@pytest.mark.parametrize("mode", ["tinyzchunk", "paragraph", "window", "document"])
@pytest.mark.parametrize("name", sorted(DOCS))
def test_chunks_are_lossless(mode: str, name: str) -> None:
    chunks = Chunker(mode=mode, max_chunk_chars=200, min_chunk_chars=20).split("d", DOCS[name])
    assert_lossless(canonicalize(DOCS[name]), chunks)


def test_crlf_chunks_identically_to_lf() -> None:
    """Line-ending normalisation makes CRLF and LF inputs equivalent."""
    unix = canonicalize(DOCS["crlf"])
    a = Chunker(mode="tinyzchunk").split("d", DOCS["crlf"])
    b = Chunker(mode="tinyzchunk").split("d", unix)
    assert [c.text for c in a] == [c.text for c in b]
    assert [(c.start, c.end) for c in a] == [(c.start, c.end) for c in b]


def test_max_chunk_chars_is_never_exceeded() -> None:
    """Regression: the upstream chunker exceeds the ceiling on a 24k one-liner."""
    for name, text in DOCS.items():
        for mode in ("tinyzchunk", "paragraph", "window"):
            chunks = Chunker(mode=mode, max_chunk_chars=300, min_chunk_chars=20).split("d", text)
            oversized = [c for c in chunks if len(c.text) > 300]
            assert not oversized, f"{name}/{mode}: {len(oversized)} oversized chunks"


def test_oversized_split_prefers_word_boundaries() -> None:
    chunks = Chunker(mode="none", max_chunk_chars=50, enforce_max_chars=True).split(
        "d", "palavra " * 40
    )
    assert all(len(c.text) <= 50 for c in chunks)
    # every chunk should still be made of whole words
    for c in chunks:
        assert c.text.split() and all(w == "palavra" for w in c.text.split())


def test_unbreakable_run_is_hard_cut_and_still_lossless() -> None:
    text = "x" * 5000
    chunks = Chunker(mode="tinyzchunk", max_chunk_chars=300).split("d", text)
    assert_lossless(text, chunks)
    assert max(len(c.text) for c in chunks) <= 300


def test_degenerate_inputs_do_not_raise() -> None:
    for text in ("", " ", "\n" * 100, "a", DOCS["one_line"], DOCS["no_spaces"]):
        Chunker(mode="tinyzchunk").split("d", text)


def test_empty_document_yields_no_chunks() -> None:
    assert Chunker(mode="tinyzchunk").split("d", "") == []


# --------------------------------------------------------------------------
# BM25: vectorised inverted index must equal the plain definition
# --------------------------------------------------------------------------
def naive_bm25(token_lists: list[list[str]], query: list[str], k1=0.8, b=0.8) -> np.ndarray:
    n = len(token_lists)
    avgdl = sum(len(t) for t in token_lists) / n
    df: dict[str, int] = {}
    for terms in token_lists:
        for t in set(terms):
            df[t] = df.get(t, 0) + 1
    scores = np.zeros(n)
    for i, terms in enumerate(token_lists):
        tf: dict[str, int] = {}
        for t in terms:
            tf[t] = tf.get(t, 0) + 1
        for q in set(query):
            if q not in df:
                continue
            idf = np.log(1.0 + (n - df[q] + 0.5) / (df[q] + 0.5))
            f = tf.get(q, 0)
            if f == 0:
                continue
            denom = f + k1 * (1.0 - b + b * len(terms) / avgdl)
            scores[i] += idf * (f * (k1 + 1.0)) / denom
    return scores


def test_bm25_matches_naive_definition() -> None:
    rng = random.Random(11)
    vocab = [f"t{i}" for i in range(30)]
    token_lists = [[rng.choice(vocab) for _ in range(rng.randint(1, 25))] for _ in range(120)]
    query = [rng.choice(vocab) for _ in range(6)]

    index = BM25Index()
    index.add(token_lists)
    index.finalize()
    got, idx = index.search(query, top_k=len(token_lists))
    want = naive_bm25(token_lists, query)

    got_full = np.zeros(len(token_lists))
    got_full[idx] = got
    assert np.allclose(got_full, want, atol=1e-5)
    assert list(idx[:10]) == list(np.argsort(-want, kind="stable")[:10])


def test_bm25_handles_repeated_terms_in_one_document() -> None:
    """Regression: duplicate term ids must not corrupt the inverted index."""
    token_lists = [["a", "a", "a", "b"], ["a", "b", "b"], ["c"]]
    index = BM25Index()
    index.add(token_lists)
    index.finalize()
    got, idx = index.search(["a"], top_k=3)
    want = naive_bm25(token_lists, ["a"])
    got_full = np.zeros(3)
    got_full[idx] = got
    assert np.allclose(got_full, want, atol=1e-6)


def test_bm25_empty_and_unknown_query() -> None:
    index = BM25Index()
    index.add([["a"], ["b"]])
    index.finalize()
    scores, idx = index.search(["zzz"], top_k=2)
    assert np.allclose(scores, 0.0)
    assert len(idx) == 2
    empty = BM25Index()
    empty.finalize()
    scores, idx = empty.search(["a"], top_k=5)
    assert len(scores) == 0 and len(idx) == 0


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------
def test_rrf_matches_definition() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60)
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62)


def test_rrf_weights_and_empty() -> None:
    assert reciprocal_rank_fusion([[], []]) == {}
    fused = reciprocal_rank_fusion([[1], [1]], k=0, weights=[1.0, 3.0])
    assert fused[1] == pytest.approx(1.0 + 3.0)


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------
def test_end_to_end_retrieval_and_offsets() -> None:
    docs = {
        "tributario": (
            "Art. 3º As faixas de alíquota do imposto de renda são progressivas.\n"
            "A alíquota máxima aplicável é de 27,5% para rendimentos acima do limite."
        ),
        "contrato": (
            "CLÁUSULA SEGUNDA - DO PAGAMENTO\n"
            "A CONTRATANTE pagará o valor mensal de R$ 10.000,00 até o quinto dia útil."
        ),
        "ambiental": (
            "A Política Nacional do Meio Ambiente institui o licenciamento ambiental "
            "como instrumento preventivo."
        ),
    }
    a = Arara(chunk_mode="tinyzchunk")
    n = a.add_documents(docs)
    a.finalize()
    assert n >= len(docs)

    hits = a.search("qual a alíquota máxima do imposto de renda?", top_k=3)
    assert hits and hits[0].doc_id == "tributario"
    for h in hits:
        assert a.resolve(h) == h.text

    assert a.search("licenciamento ambiental", top_k=1)[0].doc_id == "ambiental"


def test_crlf_document_offsets_resolve_after_canonicalisation() -> None:
    a = Arara(chunk_mode="tinyzchunk")
    a.add_documents({"crlf": "Primeira linha.\r\nSegunda linha.\r\n\r\nTerceira linha."})
    a.finalize()
    assert "\r" not in a.document_text("crlf")
    for h in a.search("segunda linha", top_k=3):
        assert a.resolve(h) == h.text


def test_rerank_orders_a_fixed_candidate_set() -> None:
    """Reranking must only reorder; the candidate set is preserved exactly."""
    a = Arara()
    a.add_documents(
        {
            "trib": "A alíquota do imposto de renda é progressiva e chega a 27,5%.",
            "amb": "O licenciamento ambiental é um instrumento preventivo.",
            "outro": "Assunto completamente diferente sobre culinária.",
        }
    )
    a.finalize()
    candidates = ["outro", "amb", "trib"]
    for mode in ("lexical", "dense", "hybrid", "cxm25"):
        ranked = a.rerank("qual a alíquota do imposto de renda?", candidates, mode=mode)
        assert ranked[0] == "trib", mode
        assert sorted(ranked) == sorted(candidates), mode


def test_rerank_puts_unscorable_documents_last() -> None:
    a = Arara()
    a.add_documents({"known": "conteúdo sobre tributos"})
    a.finalize()
    ranked = a.rerank("tributos", ["ausente", "known"], mode="lexical")
    assert ranked[-1] == "ausente"


def test_score_documents_takes_the_best_chunk() -> None:
    a = Arara(max_chunk_chars=80, min_chunk_chars=10)
    a.add_documents(
        {
            "longo": (
                "Introdução sobre um assunto qualquer que não interessa.\n\n"
                "A alíquota do imposto de renda é de vinte e sete vírgula cinco por cento.\n\n"
                "Conclusão irrelevante sobre outro tema completamente distinto."
            )
        }
    )
    a.finalize()
    scores = a.score_documents("alíquota do imposto de renda", ["longo"], mode="lexical")
    assert scores["longo"] > 0
    assert a.n_chunks > 1, "expected the document to be split"


def test_duplicate_doc_id_rejected() -> None:
    a = Arara()
    a.add_documents({"x": "texto"})
    with pytest.raises(ValueError):
        a.add_documents({"x": "outro"})


def test_incremental_add_after_finalize() -> None:
    a = Arara()
    a.add_documents({"a": "gatos são animais domésticos"})
    a.finalize()
    a.add_documents({"b": "cachorros também são animais domésticos"})
    hits = a.search("animais domésticos", top_k=2)
    assert {h.doc_id for h in hits} == {"a", "b"}


def test_search_on_empty_index() -> None:
    assert Arara().search("nada", top_k=5) == []


def test_importing_the_package_does_not_import_torch() -> None:
    """The headline claim: the runtime is torch-free."""
    import subprocess
    import sys

    code = (
        "import sys; import arara_rag; "
        "assert 'torch' not in sys.modules, 'torch was imported'; "
        "assert 'onnxruntime' not in sys.modules, 'onnxruntime was imported'; "
        "print('clean')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "clean" in out.stdout


# --------------------------------------------------------------------------
# dense encoder backends
# --------------------------------------------------------------------------
def test_encoder_registry_and_defaults() -> None:
    from arara_rag import ENCODER_BACKENDS, StaticDenseEncoder, build_encoder

    assert set(ENCODER_BACKENDS) == {"static", "use"}
    assert isinstance(build_encoder("static"), StaticDenseEncoder)
    with pytest.raises(ValueError):
        build_encoder("nope")


def test_use_encoder_is_numpy_only_and_512d() -> None:
    """The USE backend must stay torch-free and produce normalized 512-d vectors."""
    pytest.importorskip("usem3")
    from arara_rag import USEDenseEncoder

    enc = USEDenseEncoder()
    assert enc.dim == 512
    vecs = enc.encode(["o gato preto correu pelo jardim", "a menina lê um livro"])
    assert vecs.shape == (2, 512)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    import sys

    assert "torch" not in sys.modules
    assert "onnxruntime" not in sys.modules


def test_arara_accepts_dense_backend() -> None:
    pytest.importorskip("usem3")
    a = Arara(dense_backend="use")
    a.add_documents({"d": "A alíquota do imposto de renda é progressiva."})
    a.finalize()
    assert a.stats()["dense_backend"] == "use"
    assert a._dense.dim == 512
    assert a.search("alíquota do imposto", top_k=1)[0].doc_id == "d"
