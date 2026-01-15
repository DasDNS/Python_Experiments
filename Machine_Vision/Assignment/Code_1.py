import cv2
import numpy as np

# ============================================================
# NUMBER PLATE TRACKING
# ============================================================

def looks_like_text(plate_roi):
    # Edge detection to highlight text-like structures
    edges = cv2.Canny(plate_roi, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    # Horizontal projection profile
    proj = np.sum(edges, axis=0)
    thresh = np.mean(proj) * 1.5
    peaks = np.sum(proj > thresh)

    # Count contours in ROI
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Apply text-like heuristics
    if edge_density < 0.08:   
        return False
    if peaks < 3:             
        return False
    if len(cnts) < 5:         
        return False

    return True


def track_number_plate_in_video(video_path="video.mp4"):
    print("\n" + "=" * 60)
    print("CLASSICAL IMAGE PROCESSING - NUMBER PLATE TRACKING")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: could not open video.")
        return

    print("Video loaded successfully!")
    print("\nTracking plates… (Press 'q' to stop manually)\n")

    tracked_box = None  # store last detected plate

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Preprocessing: grayscale, blur, threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 165, 255, cv2.THRESH_BINARY)

        # Morphological closing to merge regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find candidate contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_boxes = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue

            aspect = w / float(h)

            # Geometric constraints for plate-like shapes
            if 2.5 < aspect < 6.0 and 100 <= w <= 220 and 25 <= h <= 65:
                plate_roi = gray[y:y+h, x:x+w]

                # Intensity constraints (black/white pixel ratios)
                black_pixels = np.mean(plate_roi < 100)
                white_pixels = np.mean(plate_roi > 150)

                if black_pixels > 0.10 and white_pixels > 0.45 and white_pixels > black_pixels:
                    if looks_like_text(plate_roi):
                        detected_boxes.append((x, y, w, h))

        # Update tracked box if new detection found
        if detected_boxes:
            tracked_box = detected_boxes[0]

        # Draw tracked plate on frame
        if tracked_box is not None:
            x, y, w, h = tracked_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, "Number Plate", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Number Plate Tracking", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("\nUser stopped the video.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nVideo processing finished!")
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    print("\nNumber Plate Detection\n")
    video = input("Enter video path (default: video.mp4): ") or "video.mp4"
    track_number_plate_in_video(video)
