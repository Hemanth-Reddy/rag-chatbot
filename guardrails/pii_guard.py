# guardrails/pii_guard.py
# Feature 2: PII Guardrails
# Scrubs personal information from user queries (before LLM)
# and LLM responses (before returning to user).

import logging
from dataclasses import dataclass, field
from typing import Optional

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)

# Every entity type we detect and replace
PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "LOCATION",
    "DATE_TIME",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "IBAN_CODE",
    "MEDICAL_LICENSE",
    "URL",
    "US_BANK_NUMBER",
    "CRYPTO",
    "NRP",
]


@dataclass
class GuardResult:
    """Structured result from one guardrail pass."""
    original_text: str
    sanitized_text: str
    pii_detected: bool
    detected_entities: list[dict] = field(default_factory=list)


class PIIGuardrail:
    """
    Bidirectional PII guardrail.

    scrub_input()  — call BEFORE sending query to LLM
    scrub_output() — call BEFORE returning LLM response to user

    Each detected entity is replaced with its type label.
    Example: "john@example.com" -> "<EMAIL_ADDRESS>"
    """

    def __init__(self, language: str = "en", score_threshold: float = 0.5):
        self.language = language
        self.score_threshold = score_threshold
        self._init_engines()

    def _init_engines(self):
        """Load spaCy NLP engine and wire up Presidio."""
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        })
        nlp_engine = provider.create_engine()

        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["en"],
        )
        self.anonymizer = AnonymizerEngine()

        # Replace every detected entity with a readable label
        self.operators = {
            entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
            for entity in PII_ENTITIES
        }
        logger.info("PIIGuardrail ready — %d entity types loaded.", len(PII_ENTITIES))

    def scrub(self, text: str) -> GuardResult:
        """Core scrub — works for any text direction."""
        if not text or not text.strip():
            return GuardResult(
                original_text=text,
                sanitized_text=text,
                pii_detected=False,
            )

        results: list[RecognizerResult] = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=PII_ENTITIES,
            score_threshold=self.score_threshold,
        )

        detected_entities = [
            {
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": round(r.score, 3),
                "original_value": text[r.start:r.end],
            }
            for r in results
        ]

        if not results:
            return GuardResult(
                original_text=text,
                sanitized_text=text,
                pii_detected=False,
                detected_entities=[],
            )

        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=self.operators,
        )

        return GuardResult(
            original_text=text,
            sanitized_text=anonymized.text,
            pii_detected=True,
            detected_entities=detected_entities,
        )

    def scrub_input(self, user_query: str) -> GuardResult:
        """Sanitize user query BEFORE sending to LLM."""
        result = self.scrub(user_query)
        if result.pii_detected:
            logger.warning(
                "PII detected in USER INPUT: %s",
                [e["entity_type"] for e in result.detected_entities],
            )
        return result

    def scrub_output(self, llm_response: str) -> GuardResult:
        """Sanitize LLM response BEFORE returning to user."""
        result = self.scrub(llm_response)
        if result.pii_detected:
            logger.warning(
                "PII detected in LLM OUTPUT: %s",
                [e["entity_type"] for e in result.detected_entities],
            )
        return result


# Module-level singleton — one instance shared across the whole app
_guardrail: Optional[PIIGuardrail] = None


def get_guardrail() -> PIIGuardrail:
    """Return the shared PIIGuardrail instance (lazy-initialized)."""
    global _guardrail
    if _guardrail is None:
        _guardrail = PIIGuardrail()
    return _guardrail