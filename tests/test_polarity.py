"""Tests for deterministic polarity detection."""

import pytest
from src.polarity import detect_polarity


class TestPositivePolarity:
    def test_supports(self):
        assert detect_polarity("WMS supports REST API") == "positive"

    def test_provides(self):
        assert detect_polarity("The platform provides real-time tracking") == "positive"

    def test_includes(self):
        assert detect_polarity("License includes 24/7 support") == "positive"

    def test_offers(self):
        assert detect_polarity("BY offers multi-tenant deployment") == "positive"

    def test_enables(self):
        assert detect_polarity("Feature enables batch processing") == "positive"

    def test_delivers(self):
        assert detect_polarity("System delivers sub-second response times") == "positive"

    def test_available(self):
        assert detect_polarity("SSO is available for enterprise customers") == "positive"

    def test_built_in(self):
        assert detect_polarity("Built-in audit logging for compliance") == "positive"

    def test_native(self):
        assert detect_polarity("Native integration with SAP") == "positive"

    def test_certified(self):
        assert detect_polarity("Certified for SOC 2 Type II") == "positive"

    def test_compliant(self):
        assert detect_polarity("Compliant with GDPR requirements") == "positive"

    def test_integrated(self):
        assert detect_polarity("Integrated with Azure AD") == "positive"


class TestNegativePolarity:
    def test_not_supported(self):
        # "not support" matches negative; "supported" does NOT match "supports?" → negative only
        assert detect_polarity("On-premise deployment is not supported") == "negative"

    def test_not_available(self):
        # "not available" negative + "available" positive → ambiguous
        assert detect_polarity("Feature not available in current release") == "unknown"

    def test_does_not(self):
        assert detect_polarity("WMS does not use Snowflake") == "negative"

    def test_cannot(self):
        assert detect_polarity("Cannot handle more than 10k SKUs") == "negative"

    def test_cant(self):
        assert detect_polarity("System can't process real-time updates") == "negative"

    def test_doesnt(self):
        # "doesn't" negative + "support" positive → ambiguous
        assert detect_polarity("Platform doesn't support multi-currency") == "unknown"

    def test_dont(self):
        # "don't" negative + "offer" positive → ambiguous
        assert detect_polarity("They don't offer SLA guarantees") == "unknown"

    def test_no_support(self):
        # "no support" negative + "support" positive → ambiguous
        assert detect_polarity("No support for legacy protocols") == "unknown"

    def test_unavailable(self):
        assert detect_polarity("Feature currently unavailable") == "negative"

    def test_excluded(self):
        assert detect_polarity("Customization is excluded from base license") == "negative"

    def test_lacks(self):
        assert detect_polarity("Product lacks real-time analytics") == "negative"

    def test_without(self):
        assert detect_polarity("Deployed without disaster recovery") == "negative"

    def test_na(self):
        # "n/a" negative + "support" positive → ambiguous
        assert detect_polarity("Multi-language support: N/A") == "unknown"

    def test_not_included(self):
        # "not included" negative; "included" does NOT match "includes?" → negative only
        assert detect_polarity("Premium features not included in starter tier") == "negative"

    def test_not_recommended(self):
        assert detect_polarity("Direct DB access not recommended") == "negative"

    def test_bare_not(self):
        assert detect_polarity("This is not a viable approach") == "negative"

    # Pure negative (no positive keyword overlap)
    def test_pure_cannot(self):
        assert detect_polarity("Cannot exceed 500 concurrent users") == "negative"

    def test_pure_excluded(self):
        assert detect_polarity("Feature is excluded from this release") == "negative"

    def test_pure_without(self):
        assert detect_polarity("Deployed without monitoring") == "negative"


class TestUnknownPolarity:
    def test_neutral_statement(self):
        assert detect_polarity("Data goes through validation") == "unknown"

    def test_empty_string(self):
        assert detect_polarity("") == "unknown"

    def test_ambiguous_both_match(self):
        # "not" matches negative, "supports" matches positive → unknown
        assert detect_polarity("WMS does not support but provides alternative") == "unknown"

    def test_plain_description(self):
        assert detect_polarity("The system uses PostgreSQL 14") == "unknown"

    def test_number_only(self):
        assert detect_polarity("Response time is 200ms") == "unknown"


class TestCaseInsensitivity:
    def test_uppercase_not(self):
        # "NOT" negative + "support" positive → ambiguous
        assert detect_polarity("WMS does NOT support legacy mode") == "unknown"

    def test_uppercase_cannot(self):
        assert detect_polarity("CANNOT process in real-time") == "negative"

    def test_mixed_case_supports(self):
        assert detect_polarity("PLATFORM SUPPORTS REST API") == "positive"

    def test_title_case(self):
        assert detect_polarity("Available For Enterprise Customers") == "positive"
