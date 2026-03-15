"""Provider abstraction for multi-model extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExtractionRequest:
    """A provider-agnostic extraction request."""

    system_prompt: str
    user_prompt: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.2
    response_format: str = "json"  # "json" | "text"


@dataclass
class ExtractionResponse:
    """A provider-agnostic extraction response."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str  # "anthropic" | "google"
    cost_estimate: float


class ExtractionProvider(ABC):
    """Base class for LLM extraction providers."""

    @abstractmethod
    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        """Send an extraction request and return structured response."""
        ...
