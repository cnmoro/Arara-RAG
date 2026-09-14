"""Gradio demo for the arara-rag HuggingFace Space.

Paste Portuguese documents, ask a question, and see which exact source span
each result points at. Runs on CPU; the whole Space needs no GPU.

Gradio is imported lazily inside :func:`build_demo` so that the parsing and
retrieval logic stays importable -- and testable -- without the UI stack
installed.

    python app.py
"""

from __future__ import annotations

from arara_rag import Arara

PLACEHOLDER_DOCS = """=== lei_1234.txt ===
Art. 1º Esta Lei estabelece as normas gerais sobre o imposto de renda das pessoas físicas.
Art. 3º As faixas de alíquota são progressivas: até R$ 2.259,20, isento; de R$ 2.259,21 a
R$ 2.826,65, 7,5%; de R$ 2.826,66 a R$ 3.751,05, 15%; acima de R$ 4.664,68, 27,5%.
Art. 4º O imposto deve ser pago até o último dia útil do mês seguinte ao do fato gerador.

=== contrato_servicos.txt ===
CLÁUSULA PRIMEIRA - DO OBJETO
O presente contrato tem por objeto a prestação de serviços de consultoria tributária.
CLÁUSULA SEGUNDA - DO PAGAMENTO
A CONTRATANTE pagará à CONTRATADA o valor mensal de R$ 10.000,00 até o quinto dia útil.
CLÁUSULA TERCEIRA - DO SIGILO
A CONTRATADA obriga-se a manter sigilo absoluto sobre as informações a que tiver acesso.

=== licenciamento_ambiental.txt ===
A Política Nacional do Meio Ambiente institui o licenciamento ambiental como instrumento
preventivo. Compete ao órgão ambiental estadual o licenciamento de empreendimentos de
impacto regional, e ao IBAMA os de impacto nacional.
"""


def parse_documents(blob: str) -> dict[str, str]:
    """Split a ``=== name ===`` delimited blob into documents."""
    docs: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    for line in blob.splitlines():
        stripped = line.strip()
        if stripped.startswith("===") and stripped.endswith("==="):
            if name is not None:
                docs[name] = "\n".join(buf).strip()
            name = stripped.strip("= ").strip() or f"doc{len(docs)}"
            buf = []
        else:
            buf.append(line)
    if name is not None:
        docs[name] = "\n".join(buf).strip()
    elif blob.strip():
        docs["documento"] = blob.strip()
    return {k: v for k, v in docs.items() if v}


def run(query: str, blob: str, mode: str = "hybrid", top_k: int = 3, chunk_mode: str = "tinyzchunk"):
    """Return ``(rows, info)`` for the UI. Pure logic, no gradio."""
    docs = parse_documents(blob or "")
    if not docs:
        return [], "Nenhum documento fornecido."
    if not query.strip():
        return [], "Escreva uma pergunta."

    arara = Arara(chunk_mode=chunk_mode)
    arara.add_documents(docs)
    arara.finalize(build_cxm25=mode == "hybrid_cxm25", n_jobs=2)
    hits = arara.search(query, top_k=int(top_k), mode=mode)

    rows = []
    for i, h in enumerate(hits, start=1):
        preview = arara.resolve(h).replace("\n", " ")
        if len(preview) > 400:
            preview = preview[:400] + "..."
        rows.append([i, round(h.score, 4), h.doc_id, f"{h.start}-{h.end}", preview])

    stats = arara.stats()
    info = (
        f"{stats['documents']} documentos → {stats['chunks']} chunks · "
        f"modo `{mode}` · chunking `{stats['chunk_mode']}` · "
        f"índice denso {stats['dense_bytes'] / 1e6:.1f} MB + "
        f"léxico {stats['lexical_bytes'] / 1e6:.1f} MB · "
        f"sem GPU, sem PyTorch"
    )
    return rows, info


def build_demo():
    """Construct the Gradio UI. Requires ``gradio``."""
    import gradio as gr

    with gr.Blocks(title="arara-rag") as demo:
        gr.Markdown(
            "# arara-rag\n"
            "**Recuperação para RAG em português, inteiramente em CPU.** Chunking, "
            "embeddings estáticos, BM25 e reranking lexical — só numpy, sem PyTorch.\n\n"
            "Cada resultado aponta o trecho exato do documento de origem, com offsets."
        )
        with gr.Row():
            with gr.Column(scale=3):
                query = gr.Textbox(label="Pergunta", value="qual a alíquota do imposto de renda?")
                blob = gr.Textbox(
                    label="Documentos (separe com === nome ===)",
                    value=PLACEHOLDER_DOCS,
                    lines=18,
                )
                with gr.Row():
                    mode = gr.Dropdown(
                        ["lexical", "hybrid", "dense", "hybrid_cxm25"],
                        value="hybrid",
                        label="Modo",
                    )
                    chunk_mode = gr.Dropdown(
                        ["tinyzchunk", "window", "paragraph", "document"],
                        value="tinyzchunk",
                        label="Chunking",
                    )
                    top_k = gr.Slider(1, 10, value=3, step=1, label="Resultados")
                go = gr.Button("Buscar", variant="primary")
            with gr.Column(scale=2):
                info = gr.Markdown()
                out = gr.Dataframe(
                    headers=["#", "score", "documento", "offsets", "trecho exato"],
                    wrap=True,
                )
        go.click(run, [query, blob, mode, top_k, chunk_mode], [out, info])
        demo.load(run, [query, blob, mode, top_k, chunk_mode], [out, info])
    return demo


if __name__ == "__main__":
    build_demo().launch()
