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
    
    Args:
        video_path: Path to video file
        output_dir: Where to save frames
        threshold: % of pixels changed to trigger capture (0.05 = 5%)
        sample_rate: Check every N seconds
    
    Returns:
        List of {"timestamp": float, "path": str}
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
    return saved_frames