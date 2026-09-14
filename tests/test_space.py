"""Tests for the demo Space logic (no gradio required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "space"))

from app import parse_documents, run  # noqa: E402


def test_parse_documents_splits_on_delimiters() -> None:
    blob = "=== a.txt ===\nconteúdo A\n\n=== b.txt ===\nconteúdo B\n"
    docs = parse_documents(blob)
    assert set(docs) == {"a.txt", "b.txt"}
    assert docs["a.txt"] == "conteúdo A"
    assert docs["b.txt"] == "conteúdo B"


def test_parse_documents_without_delimiters() -> None:
    assert parse_documents("texto solto") == {"documento": "texto solto"}
    assert parse_documents("") == {}
    assert parse_documents("   \n  ") == {}


def test_run_returns_exact_spans() -> None:
    blob = (
        "=== tributario ===\n"
        "A alíquota máxima do imposto de renda é de 27,5% para rendimentos acima do limite.\n"
        "\n=== ambiental ===\n"
        "O licenciamento ambiental é um instrumento preventivo da política de meio ambiente.\n"
    )
    rows, info = run("qual a alíquota máxima do imposto de renda?", blob, mode="lexical", top_k=2)
    assert rows, "expected at least one hit"
    assert rows[0][2] == "tributario"
    assert "27,5%" in rows[0][4]
    assert "chunks" in info

    rows, info = run("licenciamento ambiental", blob, mode="lexical", top_k=1)
    assert rows[0][2] == "ambiental"


def test_run_handles_empty_inputs() -> None:
    rows, info = run("pergunta", "", mode="hybrid")
    assert rows == []
    assert "Nenhum documento" in info

    rows, info = run("   ", "=== a ===\nalgum texto", mode="hybrid")
    assert rows == []
    assert "pergunta" in info.lower()


def test_run_hybrid_cxm25_mode() -> None:
    blob = "=== a ===\nO contrato prevê pagamento mensal de dez mil reais.\n\n=== b ===\nO imóvel fica na rua das flores."
    rows, info = run("qual o valor do pagamento mensal?", blob, mode="hybrid_cxm25", top_k=2)
    assert rows and rows[0][2] == "a"
    assert "hybrid_cxm25" in info
