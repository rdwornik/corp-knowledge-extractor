"""
Intelligent transcription provider selection with cost estimation.

Analyzes audio duration, file size, and API limits to recommend
the most cost-effective transcription provider.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config_loader import get


def get_audio_duration(file_path: str) -> float:
    """
    Get duration of audio/video file in seconds using FFprobe.

    Args:
        file_path: Path to audio/video file

    Returns:
        Duration in seconds
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise ValueError(f"Failed to get audio duration: {e}")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "5h 20min" or "45min" or "3min 25s"

    Examples:
        >>> format_duration(19213)
        '5h 20min'
        >>> format_duration(2700)
        '45min'
        >>> format_duration(205)
        '3min 25s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}min")
    if secs > 0 and hours == 0:  # Only show seconds if < 1 hour
        parts.append(f"{secs}s")

    return " ".join(parts) if parts else "0s"


def estimate_groq_availability(duration_seconds: float) -> Dict:
    """
    Estimate Groq availability and time requirements.

    Args:
        duration_seconds: Audio duration in seconds

    Returns:
        Dictionary with availability info:
        {
            "available": bool,
            "reason": str,
            "sessions_needed": int,
            "total_time_minutes": float,
            "wait_time_minutes": float
        }
    """
    rate_limit = get("settings", "transcription.groq.rate_limit_seconds_per_hour", 7200)
    max_safe_duration = get("settings", "transcription.auto_selection.groq_max_duration_seconds", 6000)

    if duration_seconds <= max_safe_duration:
        # Within safe limits
        processing_time = duration_seconds / 60 * 0.5  # Estimate: ~0.5x realtime
        return {
            "available": True,
            "reason": "Within free tier limits",
            "sessions_needed": 1,
            "total_time_minutes": processing_time,
            "wait_time_minutes": 0
        }
    else:
        # Exceeds safe limits
        sessions_needed = int((duration_seconds / rate_limit) + 1)
        wait_time_per_session = 20  # Estimate: 20 min wait between sessions
        total_wait = (sessions_needed - 1) * wait_time_per_session

        processing_time = duration_seconds / 60 * 0.5
        total_time = processing_time + total_wait

        return {
            "available": False,
            "reason": f"Exceeds hourly limit ({rate_limit}s)",
            "sessions_needed": sessions_needed,
            "total_time_minutes": total_time,
            "wait_time_minutes": total_wait
        }


def estimate_openai_cost(duration_seconds: float) -> Dict:
    """
    Estimate OpenAI Whisper API cost and time.

    Args:
        duration_seconds: Audio duration in seconds

    Returns:
        Dictionary with cost info:
        {
            "available": bool,
            "cost_usd": float,
            "time_minutes": float
        }
    """
    cost_per_minute = get("settings", "transcription.openai.cost_per_minute", 0.006)
    duration_minutes = duration_seconds / 60

    # OpenAI processes ~10x faster than realtime
    processing_time = duration_minutes * 0.1

    return {
        "available": True,
        "cost_usd": round(duration_minutes * cost_per_minute, 2),
        "time_minutes": round(processing_time, 1)
    }


def estimate_transcription(file_path: str) -> Dict:
    """
    Estimate transcription time and cost for all providers.

    Args:
        file_path: Path to audio/video file

    Returns:
        Dictionary with complete estimation:
        {
            "duration_seconds": float,
            "duration_formatted": str,
            "groq": {...},
            "openai": {...},
            "recommendation": str,
            "recommendation_reason": str
        }

    Example:
        >>> estimate = estimate_transcription("meeting.mp4")
        >>> print(f"Recommended: {estimate['recommendation']}")
        Recommended: openai
    """
    # Get duration
    duration_seconds = get_audio_duration(file_path)
    duration_formatted = format_duration(duration_seconds)

    # Estimate both providers
    groq_estimate = estimate_groq_availability(duration_seconds)
    openai_estimate = estimate_openai_cost(duration_seconds)

    # Determine recommendation
    prefer_free = get("settings", "transcription.auto_selection.prefer_free", True)

    if groq_estimate["available"] and prefer_free:
        recommendation = "groq"
        recommendation_reason = "Within free tier limits"
    elif groq_estimate["sessions_needed"] == 1 and prefer_free:
        recommendation = "groq"
        recommendation_reason = "Single session (may require one wait)"
    else:
        recommendation = "openai"
        if groq_estimate["sessions_needed"] > 1:
            recommendation_reason = (
                f"File too large for Groq free tier "
                f"(would require {groq_estimate['sessions_needed']} sessions "
                f"with {groq_estimate['wait_time_minutes']:.0f}min waits)"
            )
        else:
            recommendation_reason = "Faster processing with OpenAI"

    return {
        "duration_seconds": duration_seconds,
        "duration_formatted": duration_formatted,
        "groq": groq_estimate,
        "openai": openai_estimate,
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason
    }


def print_estimate(estimate: Dict, file_path: str):
    """
    Print formatted cost estimate to console.

    Args:
        estimate: Estimation dictionary from estimate_transcription()
        file_path: Path to file being estimated
    """
    print("\n" + "=" * 60)
    print("TRANSCRIPTION ESTIMATE")
    print("=" * 60)
    print(f"File: {os.path.basename(file_path)}")
    print(f"Duration: {estimate['duration_formatted']} ({estimate['duration_seconds']:.0f} seconds)")
    print()
    print("Provider options:")
    print()

    # Groq option
    groq = estimate["groq"]
    if groq["available"]:
        print(f"[1] Groq (FREE) - ✅ Available")
        print(f"    Total time: ~{groq['total_time_minutes']:.0f} minutes")
    else:
        print(f"[1] Groq (FREE) - ⚠️  {groq['reason']}")
        print(f"    Would require ~{groq['sessions_needed']} sessions with {groq['wait_time_minutes']:.0f}min waits")
        print(f"    Total time: ~{groq['total_time_minutes']:.0f} minutes")

    print()

    # OpenAI option
    openai = estimate["openai"]
    is_recommended = estimate["recommendation"] == "openai"
    checkmark = "✅ Recommended" if is_recommended else "Available"
    print(f"[2] OpenAI Whisper ({checkmark})")
    print(f"    Cost: ${openai['cost_usd']:.2f}")
    print(f"    Total time: ~{openai['time_minutes']:.0f} minutes")

    print()
    print("-" * 60)
    print(f"Auto-selecting: {estimate['recommendation'].upper()}")
    print(f"Reason: {estimate['recommendation_reason']}")
    print("=" * 60)
    print()


def select_provider(
    file_path: str,
    force_provider: Optional[str] = None,
    show_estimate: bool = True
) -> str:
    """
    Select transcription provider based on file analysis.

    Args:
        file_path: Path to audio/video file
        force_provider: Force specific provider ("groq", "openai")
        show_estimate: Print cost estimate to console

    Returns:
        Provider name: "groq" or "openai"

    Example:
        >>> provider = select_provider("meeting.mp4")
        === TRANSCRIPTION ESTIMATE ===
        ...
        Auto-selecting: OPENAI
        >>> print(provider)
        openai
    """
    # Check for force override
    if force_provider:
        if force_provider not in ["groq", "openai"]:
            raise ValueError(f"Invalid provider: {force_provider}. Must be 'groq' or 'openai'")
        return force_provider

    # Get estimate
    estimate = estimate_transcription(file_path)

    # Print if requested
    if show_estimate:
        print_estimate(estimate, file_path)

    return estimate["recommendation"]


def check_api_keys() -> Dict[str, bool]:
    """
    Check which API keys are available in environment.

    Returns:
        Dictionary mapping provider to availability:
        {
            "groq": bool,
            "openai": bool
        }
    """
    return {
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY"))
    }


def validate_provider_selection(provider: str) -> None:
    """
    Validate that selected provider has API key configured.

    Args:
        provider: Provider name ("groq" or "openai")

    Raises:
        ValueError: If API key not found for selected provider
    """
    api_keys = check_api_keys()

    if provider == "groq" and not api_keys["groq"]:
        raise ValueError(
            "GROQ_API_KEY not found in .env file. "
            "Please add your Groq API key or use --provider openai"
        )

    if provider == "openai" and not api_keys["openai"]:
        raise ValueError(
            "OPENAI_API_KEY not found in .env file. "
            "Please add your OpenAI API key or use --provider groq"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python provider_selector.py <audio_file>")
        print("\nExample:")
        print("  python provider_selector.py meeting.mp4")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Generate estimate
    estimate = estimate_transcription(file_path)
    print_estimate(estimate, file_path)

    # Check API keys
    api_keys = check_api_keys()
    print("\nAPI Keys configured:")
    print(f"  Groq: {'✅ Yes' if api_keys['groq'] else '❌ No'}")
    print(f"  OpenAI: {'✅ Yes' if api_keys['openai'] else '❌ No'}")
