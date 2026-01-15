import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# TEMPLATE MATCHING TOOLBOX FUNCTIONS
# ============================================================

def save_and_show_template_matching_report(frame, template, result, top_left, w, h, max_val):
    """
    Creates and saves the 4-panel template matching figure.
    """
    print("\nGenerating template matching report...")

    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw rectangle for display
    result_img = frame.copy()
    cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 3)
    cv2.putText(result_img, f"{max_val:.2%}",
                (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Template Matching Results", fontsize=16, fontweight="bold")

    # Original
    axes[0, 0].imshow(frame_rgb)
    axes[0, 0].set_title("1. Original Frame")
    axes[0, 0].axis("off")

    # Template
    axes[0, 1].imshow(template_rgb)
    axes[0, 1].set_title("2. Template to Find")
    axes[0, 1].axis("off")

    # Heatmap
    axes[1, 0].imshow(result, cmap='hot')
    axes[1, 0].set_title("3. Matching Heatmap")
    axes[1, 0].plot(top_left[0], top_left[1], 'b*', markersize=20)
    axes[1, 0].axis("off")

    # Detection
    axes[1, 1].imshow(result_rgb)
    axes[1, 1].set_title(f"4. Detection Result\nConfidence: {max_val:.2%}")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("template_matching_result.png", dpi=150, bbox_inches="tight")
    print("Saved 'template_matching_result.png'")
    plt.show()


def save_and_show_all_methods_comparison(frame, template_gray):
    """
    Creates the 'all methods comparison' multi-plot.
    """
    print("Generating comparison of all template matching methods...")

    h, w = template_gray.shape
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    methods = {
        'TM_CCOEFF': cv2.TM_CCOEFF,
        'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
        'TM_CCORR': cv2.TM_CCORR,
        'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
        'TM_SQDIFF': cv2.TM_SQDIFF,
        'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.ravel()
    fig.suptitle("Comparison of All Matching Methods", fontsize=16, fontweight="bold")

    for idx, (method_name, method) in enumerate(methods.items()):
        result = cv2.matchTemplate(frame_gray, template_gray, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            score = min_val
            score_type = "Min"
        else:
            top_left = max_loc
            score = max_val
            score_type = "Max"

        img_copy = frame.copy()
        br = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img_copy, top_left, br, (0, 255, 0), 3)
        rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

        axes[idx].imshow(rgb)
        axes[idx].set_title(f"{method_name}\n{score_type}: {score:.4f}")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig("all_methods_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved 'all_methods_comparison.png'")
    plt.show()


# ============================================================
# VIDEO TEMPLATE MATCHING PIPELINE
# ============================================================

def track_number_plate_in_video(video_path='video.mp4', template_path='plate_template.png'):
    print("\n" + "=" * 60)
    print("VIDEO TEMPLATE MATCHING - NUMBER PLATE TRACKING")
    print("=" * 60)

    # Load template
    print("\nLoading template...")
    template = cv2.imread(template_path)
    if template is None:
        print(f"Cannot load template: {template_path}")
        return

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape
    print(f"Template loaded ({w}x{h})")

    # Open video
    print("\nOpening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    print("Tracking started. Press 'q' to quit manually.\n")

    # Track best detection across the entire video
    last_frame = None
    last_result = None
    last_top_left = None
    last_max_val = 0
    detection_threshold = 0.55  # confidence threshold for live visualization

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Draw rectangle in live video if above threshold
        if max_val > detection_threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
            cv2.putText(frame, f"{max_val:.2%}", (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Store best detection for end-of-video results
        if max_val > last_max_val:
            last_frame = frame.copy()
            last_result = result.copy()
            last_top_left = max_loc
            last_max_val = max_val

        # Show live video
        cv2.imshow("Number Plate Tracking", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("\nManually stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if last_frame is None:
        print("No detection during video. Ending.")
        return

    print(f"\nDetection confidence: {last_max_val:.2%}")

    # Save detection image
    br = (last_top_left[0] + w, last_top_left[1] + h)
    detection_img = last_frame.copy()
    cv2.rectangle(detection_img, last_top_left, br, (0, 255, 0), 3)
    cv2.imwrite("number_plate_detection_result.png", detection_img)
    print("Saved: number_plate_detection_result.png")

    # Show final reports automatically
    save_and_show_template_matching_report(
        last_frame, template, last_result,
        last_top_left, w, h, last_max_val
    )

    save_and_show_all_methods_comparison(last_frame, template_gray)

    print("\nVideo tracking complete!")
    print("=" * 60)


# ============================================================
# MAIN FLOW
# ============================================================

if __name__ == "__main__":
    
    print("Number Plate Tracking System\n")

    choice = input("Start video-based template matching? (yes/no): ")

    if choice.lower() == 'yes':
        video = input("Enter video path (default: video.mp4): ") or "video.mp4"
        temp = input("Enter template path (.png/.jpg): ") or "plate_template.png"
        track_number_plate_in_video(video, temp)

    print("\nProgram completed successfully!")

