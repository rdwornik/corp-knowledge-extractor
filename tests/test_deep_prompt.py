"""Tests for deep prompt builder."""

import pytest
from src.deep_prompt import build_deep_prompt


def test_build_deep_prompt_architecture():
    """Architecture deep prompt includes overlay fields."""
    prompt = build_deep_prompt("architecture")
    assert "architecture" in prompt
    assert "components" in prompt
    assert "integration_points" in prompt
    assert "deployment_model" in prompt
    assert "tech_stack" in prompt


def test_build_deep_prompt_security():
    """Security deep prompt includes security-specific overlay fields."""
    prompt = build_deep_prompt("security")
    assert "security" in prompt
    assert "certifications" in prompt
    assert "encryption" in prompt
    assert "dr_rto" in prompt
    assert "dr_rpo" in prompt


def test_build_deep_prompt_commercial():
    """Commercial deep prompt includes pricing/SLA overlay fields."""
    prompt = build_deep_prompt("commercial")
    assert "commercial" in prompt
    assert "pricing_model" in prompt
    assert "sla_tiers" in prompt
    assert "support_tiers" in prompt


def test_build_deep_prompt_rfp_response():
    """RFP response deep prompt includes Q&A overlay fields."""
    prompt = build_deep_prompt("rfp_response")
    assert "rfp_response" in prompt
    assert "questions_answered" in prompt
    assert "capabilities_demonstrated" in prompt
    assert "gaps_identified" in prompt


def test_build_deep_prompt_meeting():
    """Meeting deep prompt includes attendees/action items overlay fields."""
    prompt = build_deep_prompt("meeting")
    assert "meeting" in prompt
    assert "attendees" in prompt
    assert "decisions_made" in prompt
    assert "action_items" in prompt


def test_build_deep_prompt_contains_base_structure():
    """All deep prompts contain the base extraction structure."""
    prompt = build_deep_prompt("architecture")
    assert "key_facts" in prompt
    assert "entities_mentioned" in prompt
    assert "extraction_version" in prompt
    assert '"base"' in prompt
    assert '"overlay"' in prompt


def test_build_deep_prompt_unknown_type_still_works():
    """Unknown doc_type builds a prompt with empty overlay."""
    prompt = build_deep_prompt("unknown_type")
    assert "unknown_type" in prompt
    assert '"base"' in prompt
