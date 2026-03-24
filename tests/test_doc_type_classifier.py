"""Tests for the deterministic document type classifier."""

import pytest
from src.doc_type_classifier import (
    classify_doc_type,
    classify_from_filename,
    should_extract_deep,
    DEEP_DOC_TYPES,
)


class TestFolderPathRules:
    """Folder path patterns are the most reliable classification signal."""

    def test_product_docs_folder(self):
        assert classify_doc_type("C:/MyWork/01_Product_Docs/WMS_Overview.pdf") == "product_doc"

    def test_product_docs_folder_generic(self):
        assert classify_doc_type("/data/product_docs/Platform.pptx") == "product_doc"

    def test_competitive_folder(self):
        assert classify_doc_type("C:/MyWork/03_Competitive/Analysis.xlsx") == "commercial"

    def test_training_folder_numbered(self):
        assert classify_doc_type("C:/MyWork/02_Training/Bootcamp.pptx") == "training"

    def test_training_folder_generic(self):
        assert classify_doc_type("/data/training/session1.pdf") == "training"

    def test_certificate_folder(self):
        assert classify_doc_type("C:/50_RFP/Certificate/SOC2.pdf") == "security"

    def test_security_folder(self):
        assert classify_doc_type("/docs/security/whitepaper.pdf") == "security"

    def test_compliance_folder(self):
        assert classify_doc_type("/docs/compliance/gdpr_report.pdf") == "security"

    def test_rfp_response_folder(self):
        assert classify_doc_type("C:/50_RFP/Response/client_rfp.docx") == "rfp_response"

    def test_rfp_answer_folder(self):
        assert classify_doc_type("C:/RFP/Answer/response.docx") == "rfp_response"

    def test_discovery_folder(self):
        assert classify_doc_type("C:/Projects/Discovery/initial_call.docx") == "meeting"

    def test_meeting_folder(self):
        assert classify_doc_type("C:/Projects/Meeting/notes_2026.md") == "meeting"

    def test_workshop_folder(self):
        assert classify_doc_type("C:/Projects/Workshop/day1.pptx") == "meeting"


class TestFilenameRules:
    """Filename patterns are fallback when folder path doesn't match."""

    def test_architecture_filename(self):
        assert classify_doc_type("/generic/BYPlatform-Architecture.pdf") == "architecture"

    def test_platform_filename(self):
        assert classify_doc_type("/generic/Platform_Overview.pptx") == "architecture"

    def test_technical_filename(self):
        assert classify_doc_type("/generic/Technical_Deep_Dive.pdf") == "architecture"

    def test_sla_filename(self):
        assert classify_doc_type("/generic/SLA_Appendix.pdf") == "security"

    def test_security_whitepaper_filename(self):
        assert classify_doc_type("/generic/Security_Whitepaper.pdf") == "security"

    def test_iso27001_filename(self):
        assert classify_doc_type("/generic/ISO_27001_cert.pdf") == "security"

    def test_pricing_filename(self):
        assert classify_doc_type("/generic/Pricing_Sheet_2026.xlsx") == "commercial"

    def test_service_description_filename(self):
        assert classify_doc_type("/generic/WMS_Service_Description.pdf") == "commercial"

    def test_contract_filename(self):
        assert classify_doc_type("/generic/Master_Contract.pdf") == "commercial"

    def test_rfp_response_filename(self):
        assert classify_doc_type("/generic/RFP_Response_v3.docx") == "rfp_response"

    def test_meeting_notes_filename(self):
        assert classify_doc_type("/generic/Meeting_Notes_March.md") == "meeting"

    def test_training_filename(self):
        assert classify_doc_type("/generic/Enablement_Session.pptx") == "training"


class TestDefaultFallback:
    """Files that don't match any pattern get 'general'."""

    def test_random_presentation(self):
        assert classify_doc_type("/generic/quarterly_update.pptx") == "general"

    def test_random_pdf(self):
        assert classify_doc_type("/downloads/report_2026.pdf") == "general"


class TestBackslashHandling:
    """Windows backslash paths are normalized."""

    def test_windows_path(self):
        assert classify_doc_type("C:\\Users\\MyWork\\01_Product_Docs\\WMS.pdf") == "product_doc"


class TestShouldExtractDeep:
    """should_extract_deep returns True for deep doc_types."""

    def test_deep_types(self):
        for dt in DEEP_DOC_TYPES:
            assert should_extract_deep(dt) is True

    def test_general_is_standard(self):
        assert should_extract_deep("general") is False

    def test_unknown_is_standard(self):
        assert should_extract_deep("unknown") is False


# ---------------------------------------------------------------------------
# Filename-based doc_type pre-classification (BUG 2: JLR pilot)
# ---------------------------------------------------------------------------


class TestFilenameDocType:
    """Filename patterns detect RFI/RFP/Questionnaire BEFORE folder/content rules."""

    def test_filename_rfi(self):
        assert classify_from_filename("JLR TMS RFI Response.docx") == "rfp_response"

    def test_filename_questionnaire(self):
        assert classify_from_filename("VA Questionnaire.xlsx") == "vendor_assessment"

    def test_filename_rfp(self):
        assert classify_from_filename("Honda RFP Response v2.docx") == "rfp_response"

    def test_filename_discovery(self):
        assert classify_from_filename("GCC Discovery Questions.xlsx") == "discovery"

    def test_filename_no_match(self):
        assert classify_from_filename("Platform Architecture.pdf") is None

    def test_filename_case_insensitive(self):
        assert classify_from_filename("jlr_tms_rfi.docx") == "rfp_response"

    def test_filename_overrides_general_in_classify(self):
        """RFI in filename → rfp_response even in generic folder."""
        assert classify_doc_type("/generic/JLR TMS RFI Response.docx") == "rfp_response"

    def test_vendor_assessment_deep(self):
        assert should_extract_deep("vendor_assessment") is True

    def test_discovery_deep(self):
        assert should_extract_deep("discovery") is True
