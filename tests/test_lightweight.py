"""Small offline tests for stable retrieval and citation contracts."""
from dataclasses import dataclass, field

import pytest

from src.Config.settings import NateqSettings
from src.application.citation_validation import validate_structural_citations
from src.retrival.retrival import _quota_union
from src.retrival.focused_parent import select_focused_parents


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)


def test_default_settings_match_phase_8():
    settings = NateqSettings()
    assert (settings.dense_k, settings.arabic_quota) == (20, 5)
    assert (settings.retrieval_weight, settings.ce_weight) == (0.35, 0.65)
    assert settings.focus_budget_chars == 4500


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        NateqSettings(dense_k=0)
    with pytest.raises(ValueError):
        NateqSettings(retrieval_weight=0.1, ce_weight=0.1)


def test_quota_union_deduplicates_children():
    docs = [FakeDocument("alpha", {"child_id": "a"}),
            FakeDocument("beta", {"child_id": "b"}),
            FakeDocument("gamma", {"child_id": "c"})]
    selected = _quota_union(docs[:2], [docs[1], docs[2]], dense_quota=2, total=3)
    assert [doc.metadata["child_id"] for doc in selected] == ["a", "b", "c"]


def test_focused_window_keeps_exact_child_and_page():
    parent_text = "A" * 300 + "RETRIEVED HISTORICAL EVIDENCE" + "Z" * 350
    parent = FakeDocument(parent_text, {"parent_id": "p1", "page_number": 15})
    child = FakeDocument("المصدر: History\nرقم الصفحة: 15\n\nRETRIEVED HISTORICAL EVIDENCE",
                         {"parent_id": "p1", "child_id": "p1_c0", "contextual_header": True})
    docs, diagnostics = select_focused_parents([child], [parent], focus_budget_chars=450, min_chars=100)
    assert docs[0].metadata["page_number"] == 15
    assert "RETRIEVED HISTORICAL EVIDENCE" in docs[0].page_content
    assert len(docs[0].page_content) <= 450
    assert diagnostics[0]["child_text_present"]


def test_structural_citation_check():
    result = {"status": "answered", "citations": ["E1"], "sources": [
        {"reference": "E1", "content": "A quoted source passage", "page_number": 15}
    ]}
    assert validate_structural_citations(result)["valid"]
    assert not validate_structural_citations({**result, "citations": ["E2"]})["valid"]
    assert not validate_structural_citations({**result, "citations": []})["valid"]
    assert validate_structural_citations({**result, "status": "insufficient_evidence", "citations": []})["valid"]


def test_cited_sources_show_only_references_used_in_answer():
    from src.application.source_display import build_cited_sources, format_answer_with_sources

    sources = [
        {
            "reference": "E1", "page_number": 9,
            "content": "انتقل من تكريت إلى الموصل مع والده.",
            "metadata": {
                "document_id": "salah_al_din",
                "document_title": "سيرة صلاح الدين الأيوبي",
                "source": r"D:\\2-NATEQ\\data\\salah_al_din.pdf",
                "page_number": 9,
            },
        },
        {
            "reference": "E2", "page_number": 15,
            "content": "نص آخر غير مستشهد به",
            "metadata": {"document_title": "سيرة صلاح الدين الأيوبي", "page_number": 15},
        },
    ]
    cited = build_cited_sources(["E1"], sources)
    assert len(cited) == 1
    assert cited[0]["reference"] == "E1"
    assert cited[0]["title"] == "سيرة صلاح الدين الأيوبي"
    assert cited[0]["page_number"] == 9
    assert cited[0]["file_name"] == "salah_al_din.pdf"
    assert "D:" not in str(cited)
    displayed = format_answer_with_sources("انتقل إلى الموصل.", cited)
    assert "المصادر:" in displayed
    assert "صفحة 9" in displayed
    assert "انتقل من تكريت إلى الموصل" in displayed
    assert "صفحة 15" not in displayed


def test_no_citations_means_no_displayed_sources():
    from src.application.source_display import build_cited_sources, format_answer_with_sources

    assert build_cited_sources([], [{"reference": "E1", "content": "x"}]) == []
    assert format_answer_with_sources("لا تتوفر معلومات كافية.", []) == "لا تتوفر معلومات كافية."


def test_source_preview_is_shorter_but_full_content_remains():
    from src.application.source_display import build_cited_sources

    text = "historical source " * 100
    displayed = build_cited_sources(
        ["E1"],
        [{"reference": "E1", "content": text, "page_number": 12, "metadata": {}}],
        preview_chars=40,
    )
    assert len(displayed[0]["excerpt"]) <= 41
    assert displayed[0]["content"] == text.strip()


def test_ask_exposes_user_facing_sources_after_structural_check():
    """Test the response contract without loading Chroma, Groq, or embedding models."""
    from src.application.rag_service_provider import RAGService

    class MinimalRAG(RAGService):
        def __init__(self, is_valid=True):
            self.is_valid = is_valid

        def focused_parent_retrieval(self, question):
            doc = FakeDocument(
                "ثم اتفق له الانتقال من تكريت إلى الموصل المحروسة.",
                {
                    "page_number": 9,
                    "document_title": "سيرة صلاح الدين الأيوبي",
                    "source": r"D:\2-NATEQ\data\salah_al_din.pdf",
                    "document_id": "salah_al_din",
                },
            )
            diagnostics = {
                "retrieval_seconds": 0.1,
                "rerank_seconds": 0.1,
                "context_chars": len(doc.page_content),
            }
            return [doc], diagnostics

        def generate(self, question, documents):
            return {
                "status": "answered",
                "answer": "انتقل إلى الموصل.",
                "citations": ["E1" if self.is_valid else "E99"],
            }

    response = MinimalRAG().ask("أين انتقل بعد تكريت؟")
    assert response["citation_audit"]["valid"]
    assert response["citations"] == ["E1"]
    assert len(response["cited_sources"]) == 1
    assert response["cited_sources"][0]["page_number"] == 9
    assert "سيرة صلاح الدين الأيوبي" in response["answer_with_sources"]
    assert "الموصل" in response["answer_with_sources"]

    invalid = MinimalRAG(is_valid=False).ask("أين انتقل بعد تكريت؟")
    assert invalid["status"] == "insufficient_evidence"
    assert invalid["cited_sources"] == []
    assert "المصادر:" not in invalid["answer_with_sources"]
