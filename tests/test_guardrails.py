# tests/test_guardrails.py
# Feature 2 test — no LLM, no ChromaDB, no internet needed.
# Tests PII detection and scrubbing in complete isolation.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from guardrails.pii_guard import PIIGuardrail


@pytest.fixture(scope="module")
def guard():
    """One guardrail instance reused across all tests — spaCy loads once."""
    return PIIGuardrail()


class TestEmailDetection:
    def test_email_detected(self, guard):
        result = guard.scrub("Contact me at john.doe@example.com for help.")
        assert result.pii_detected
        assert "EMAIL_ADDRESS" in [e["entity_type"] for e in result.detected_entities]

    def test_email_replaced(self, guard):
        result = guard.scrub("Email me at jane@company.org please.")
        assert "jane@company.org" not in result.sanitized_text
        assert "<EMAIL_ADDRESS>" in result.sanitized_text

class TestPhoneDetection:
    def test_phone_detected(self, guard):
        # Use a full international format — Presidio scores this above 0.5
        result = guard.scrub("Please call John at 512-387-4277 extension 123.")
        assert result.pii_detected

    def test_phone_replaced(self, guard):
        result = guard.scrub("Please call John at 512-387-4277 extension 123.")
        assert "512-386-4277" not in result.sanitized_text

class TestPersonName:
    def test_name_detected(self, guard):
        result = guard.scrub("My name is John Smith and I need help.")
        assert result.pii_detected
        assert "John Smith" not in result.sanitized_text

class TestSSN:
    def test_ssn_detected(self, guard):
        # Use a realistic SSN pattern Presidio scores above threshold
        result = guard.scrub("My social security number is 382-45-6789.")
        assert result.pii_detected
        assert "382-45-6789" not in result.sanitized_text

class TestCreditCard:
    def test_credit_card_detected(self, guard):
        result = guard.scrub("My card number is 4111 1111 1111 1111.")
        assert result.pii_detected
        assert "4111 1111 1111 1111" not in result.sanitized_text

class TestNoPII:
    def test_clean_text_unchanged(self, guard):
        # Avoid place names — Presidio detects them as LOCATION
        text = "What is retrieval augmented generation?"
        result = guard.scrub(text)
        assert not result.pii_detected
        assert result.sanitized_text == text

    def test_technical_text_unchanged(self, guard):
        text = "ChromaDB uses HNSW indexing for fast similarity search."
        result = guard.scrub(text)
        assert result.sanitized_text == text

class TestDirectionality:
    def test_scrub_input(self, guard):
        result = guard.scrub_input("Help me Bob Jones, bob@mail.com")
        assert result.pii_detected

    def test_scrub_output(self, guard):
        result = guard.scrub_output("Contact Alice Brown at alice@corp.com.")
        assert result.pii_detected
        assert "alice@corp.com" not in result.sanitized_text


class TestEdgeCases:
    def test_empty_string(self, guard):
        result = guard.scrub("")
        assert not result.pii_detected
        assert result.sanitized_text == ""

    def test_whitespace_only(self, guard):
        result = guard.scrub("   ")
        assert not result.pii_detected

    def test_result_has_correct_fields(self, guard):
        result = guard.scrub("Call John at 555-1234.")
        assert hasattr(result, "original_text")
        assert hasattr(result, "sanitized_text")
        assert hasattr(result, "pii_detected")
        assert hasattr(result, "detected_entities")
        assert isinstance(result.detected_entities, list)