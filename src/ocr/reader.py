import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


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