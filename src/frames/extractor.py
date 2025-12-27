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
    
    return saved_frames


def _deduplicate_frames(frames: list[dict], threshold: float = 0.92) -> list[dict]:
    """Remove frames that are too similar to previous ones."""
    if len(frames) <= 1:
        return frames
    
    unique = [frames[0]]
    prev_img = cv2.imread(frames[0]["path"], cv2.IMREAD_GRAYSCALE)
    
    for frame in frames[1:]:
        curr_img = cv2.imread(frame["path"], cv2.IMREAD_GRAYSCALE)
        
        # Resize for faster comparison
        prev_small = cv2.resize(prev_img, (100, 100))
        curr_small = cv2.resize(curr_img, (100, 100))
        
        # Calculate similarity
        diff = cv2.absdiff(prev_small, curr_small)
        similarity = 1 - (diff.sum() / (255 * 100 * 100))
        
        if similarity < threshold:
            unique.append(frame)
            prev_img = curr_img
        else:
            # Delete duplicate file
            os.remove(frame["path"])
    
    return unique