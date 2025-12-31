import pytesseract
from PIL import Image
from config.config_loader import get

# Load tesseract path from config
tesseract_path = get("settings", "tools.tesseract_path")
pytesseract.pytesseract.tesseract_cmd = tesseract_path


def read_frame(image_path: str) -> str:
    """
    Extract text from a single frame/image.
    
    Returns:
        Extracted text as string.
    """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text.strip()


def read_frames(frames: list[dict]) -> list[dict]:
    """
    Extract text from multiple frames.
    
    Args:
        frames: List of {"timestamp": float, "path": str}
    
    Returns:
        List of {"timestamp": float, "path": str, "text": str}
    """
    results = []
    for frame in frames:
        text = read_frame(frame["path"])
        results.append({
            "timestamp": frame["timestamp"],
            "path": frame["path"],
            "text": text
        })
    return results