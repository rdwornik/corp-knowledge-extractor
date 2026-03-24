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
        assert classify_doc_type("C:/Projects/Workshop/day1.pptx") == "workshop"


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
        """'Platform Architecture.pdf' now matches architecture pattern."""
        assert classify_from_filename("Platform Architecture.pdf") == "architecture"

    def test_filename_truly_no_match(self):
        assert classify_from_filename("quarterly_update.pdf") is None

    def test_filename_case_insensitive(self):
        assert classify_from_filename("jlr_tms_rfi.docx") == "rfp_response"

    def test_filename_overrides_general_in_classify(self):
        """RFI in filename → rfp_response even in generic folder."""
        assert classify_doc_type("/generic/JLR TMS RFI Response.docx") == "rfp_response"

    def test_vendor_assessment_deep(self):
        assert should_extract_deep("vendor_assessment") is True

    def test_discovery_deep(self):
        assert should_extract_deep("discovery") is True


# ---------------------------------------------------------------------------
# Expanded classifier patterns (FIX 3: v3 regression — general catch-all)
# ---------------------------------------------------------------------------


class TestExpandedFilenamePatterns:
    """New filename patterns reduce 'general' catch-all."""

    def test_classify_frs(self):
        assert classify_from_filename("Lenzing_BSC_FRS_v2.xlsx") == "requirements_spec"

    def test_classify_functional_requirements(self):
        assert classify_from_filename("Functional_Requirements_Spec.docx") == "requirements_spec"

    def test_classify_user_stories(self):
        assert classify_from_filename("BSC_User_Stories.xlsx") == "requirements_spec"

    def test_classify_annual_report(self):
        assert classify_from_filename("Annual_Report_2023.pdf") == "financial_report"

    def test_classify_earnings(self):
        assert classify_from_filename("Q3_Earnings_Release.pdf") == "financial_report"

    def test_classify_sow(self):
        assert classify_from_filename("PoC_Statement_of_Work.docx") == "proposal"

    def test_classify_poc_proposal(self):
        assert classify_from_filename("PoC_Proposal_Lenzing.docx") == "proposal"

    def test_classify_catalog(self):
        assert classify_from_filename("Product_Catalog_Master.xlsx") == "master_data"

    def test_classify_hierarchy(self):
        assert classify_from_filename("Product_Hierarchy_v2.xlsx") == "master_data"

    def test_classify_item_master(self):
        assert classify_from_filename("Item_Master_Data.csv") == "master_data"

    def test_classify_architecture(self):
        assert classify_from_filename("Platform_Architecture_Overview.pdf") == "architecture"

    def test_classify_training(self):
        assert classify_from_filename("WMS_Training_Module_3.pptx") == "training"

    def test_classify_competitive(self):
        assert classify_from_filename("Kinaxis_vs_BY_Battlecard.pptx") == "competitive"

    def test_classify_workshop(self):
        assert classify_from_filename("Demand_Planning_Workshop.pptx") == "workshop"

    def test_classify_demo(self):
        assert classify_from_filename("WMS_Demo_Script.pptx") == "demo"

    def test_classify_meeting_notes(self):
        assert classify_from_filename("Discovery_Meeting_Notes_March.md") == "meeting"

    def test_classify_debrief(self):
        assert classify_from_filename("Design_Thinking_Debrief.docx") == "meeting"

    def test_classify_general_fallback(self):
        """Files that match nothing still get 'general'."""
        assert classify_from_filename("random_notes.txt") is None


class TestExpandedDocTypeDepth:
    """New doc_types correctly route to deep or standard."""

    def test_requirements_spec_deep(self):
        assert should_extract_deep("requirements_spec") is True

    def test_financial_report_deep(self):
        assert should_extract_deep("financial_report") is True

    def test_proposal_deep(self):
        assert should_extract_deep("proposal") is True

    def test_competitive_deep(self):
        assert should_extract_deep("competitive") is True

    def test_workshop_deep(self):
        assert should_extract_deep("workshop") is True

    def test_demo_deep(self):
        assert should_extract_deep("demo") is True

    def test_master_data_standard(self):
        """Master data is Tier 1 — no deep extraction needed."""
        assert should_extract_deep("master_data") is False


class TestExpandedFullClassification:
    """End-to-end classify_doc_type with new patterns."""

    def test_frs_in_generic_folder(self):
        assert classify_doc_type("/generic/Lenzing_BSC_FRS_v2.xlsx") == "requirements_spec"

    def test_annual_report_in_generic_folder(self):
        assert classify_doc_type("/generic/Annual_Report_2023.pdf") == "financial_report"

    def test_sow_in_generic_folder(self):
        assert classify_doc_type("/generic/PoC_Statement_of_Work.docx") == "proposal"

    def test_catalog_in_generic_folder(self):
        assert classify_doc_type("/generic/Product_Catalog_Master.xlsx") == "master_data"

    def test_workshop_folder_override(self):
        """Workshop folder → 'workshop' (was 'meeting' before)."""
        assert classify_doc_type("C:/Projects/Workshop/day1.pptx") == "workshop"

    def test_battlecard_filename(self):
        assert classify_doc_type("/generic/competitive_battlecard_kinaxis.pptx") == "competitive"
