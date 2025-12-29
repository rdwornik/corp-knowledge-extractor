import cv2
import os


def extract_frames(
    video_path: str,
    output_dir: str = "data/frames",
    threshold: float = 0.05,
    sample_rate: int = 1
) -> list[dict]:
    """
    Extract frames when slide changes detected.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * sample_rate)
    
    prev_frame = None
    saved_frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is None:
                save = True
            else:
                diff = cv2.absdiff(gray, prev_frame)
                changed = (diff > 25).sum() / diff.size
                save = changed > threshold
            
            if save:
                timestamp = frame_count / fps
                filename = f"frame_{timestamp:.1f}s.png"
                path = os.path.join(output_dir, filename)
                cv2.imwrite(path, frame)
                saved_frames.append({"timestamp": timestamp, "path": path})
                prev_frame = gray
        
        frame_count += 1
    
    cap.release()
    
    # Deduplicate similar frames
    saved_frames = _deduplicate_frames(saved_frames)
    
    # Filter out junk frames (Teams UI, waiting screens)
    saved_frames = _filter_junk_frames(saved_frames)

    # Sort by timestamp
    saved_frames = sorted(saved_frames, key=lambda x: x["timestamp"])

    return saved_frames


def _deduplicate_frames(frames: list[dict], similarity_threshold: float = 0.85) -> list[dict]:
    """Remove frames that are too similar (pixel or OCR text)."""
    import pytesseract
    from PIL import Image
    
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    if len(frames) <= 1:
        return frames
    
    unique = [frames[0]]
    prev_img = cv2.imread(frames[0]["path"], cv2.IMREAD_GRAYSCALE)
    prev_text = ""
    
    try:
        prev_text = pytesseract.image_to_string(Image.open(frames[0]["path"])).strip()
    except:
        pass
    
    for frame in frames[1:]:
        curr_img = cv2.imread(frame["path"], cv2.IMREAD_GRAYSCALE)
        curr_text = ""
        
        try:
            curr_text = pytesseract.image_to_string(Image.open(frame["path"])).strip()
        except:
            pass
        
        # Check pixel similarity
        prev_small = cv2.resize(prev_img, (100, 100))
        curr_small = cv2.resize(curr_img, (100, 100))
        diff = cv2.absdiff(prev_small, curr_small)
        pixel_similarity = 1 - (diff.sum() / (255 * 100 * 100))
        
        # Check OCR text similarity
        text_similarity = _text_similarity(prev_text, curr_text)
        
        # Keep if BOTH are different enough
        if pixel_similarity < similarity_threshold or text_similarity < 0.90:
            unique.append(frame)
            prev_img = curr_img
            prev_text = curr_text
        else:
            os.remove(frame["path"])
    
    return unique


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate word overlap similarity."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    overlap = len(words1 & words2)
    total = len(words1 | words2)
    
    return overlap / total if total > 0 else 0.0

def _filter_junk_frames(frames: list[dict]) -> list[dict]:
    """Remove frames with Teams UI, waiting screens, etc."""
    import pytesseract
    from PIL import Image
    
    # Set tesseract path
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    junk_patterns = [
        "waiting for others",
        "joining",
        "click to add subtitle",
        "microsoft teams",
        "mute",
        "leave"
    ]
    
    filtered = []
    for frame in frames:
        try:
            img = Image.open(frame["path"])
            text = pytesseract.image_to_string(img).lower()
            
            is_junk = any(pattern in text for pattern in junk_patterns)
            
            if not is_junk:
                filtered.append(frame)
            else:
                os.remove(frame["path"])
        except:
            filtered.append(frame)  # Keep if can't read
    
    return filtered