"""Tests for extended polarity classification."""

from src.polarity import classify_fact_polarity, classify_note_polarity


class TestClassifyFactPolarity:
    def test_polarity_positive(self):
        assert classify_fact_polarity("11% improvement in accuracy") == "positive"

    def test_polarity_negative(self):
        assert classify_fact_polarity("customers fear migration risk") == "negative"

    def test_polarity_mixed(self):
        assert classify_fact_polarity("improved accuracy but increased complexity and risk") == "mixed"

    def test_polarity_neutral(self):
        assert classify_fact_polarity("1706 customers identified") == "neutral"

    def test_polarity_negation(self):
        assert classify_fact_polarity("does NOT support SOAP") == "negative"


class TestClassifyNotePolarity:
    def test_note_polarity_dominant(self):
        """5 positive + 1 negative → positive (ratio > 2:1)."""
        facts = [
            {"polarity": "positive"}, {"polarity": "positive"},
            {"polarity": "positive"}, {"polarity": "positive"},
            {"polarity": "positive"}, {"polarity": "negative"},
        ]
        assert classify_note_polarity(facts) == "positive"

    def test_note_polarity_mixed(self):
        """3 positive + 3 negative → mixed."""
        facts = [
            {"polarity": "positive"}, {"polarity": "positive"},
            {"polarity": "positive"}, {"polarity": "negative"},
            {"polarity": "negative"}, {"polarity": "negative"},
        ]
        assert classify_note_polarity(facts) == "mixed"

    def test_note_polarity_empty(self):
        assert classify_note_polarity([]) == "neutral"
