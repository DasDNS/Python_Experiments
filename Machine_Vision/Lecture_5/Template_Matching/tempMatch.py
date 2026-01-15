import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# TEMPLATE MATCHING - CRICKET PLAYERS DETECTION
# ============================================================
# This script demonstrates template matching to find a face
# (Image 2) within a larger image (Image 1)
# ============================================================

def template_matching_demo():
    """
    Complete template matching demonstration with visualization
    """
    
    print("=" * 60)
    print("TEMPLATE MATCHING DEMONSTRATION")
    print("=" * 60)
    
    # ========================================
    # STEP 1: LOAD IMAGES
    # ========================================
    print("\n[1] Loading images...")
    
    # Load the source image (main image with both players)
    source_image = cv2.imread('doniNkoli.jpg')  # Change to your image path
    
    # Load the template (face to search for)
    template_image = cv2.imread('koli.jpg')  # Change to your template path
    
    # Check if images loaded successfully
    if source_image is None:
        print("❌ Error: Could not load source image!")
        print("Please update 'image1.jpg' with your actual file path")
        return
    
    if template_image is None:
        print("❌ Error: Could not load template image!")
        print("Please update 'image2.jpg' with your actual file path")
        return
    
    print(f"✓ Source image loaded: {source_image.shape}")
    print(f"✓ Template loaded: {template_image.shape}")
    
    # ========================================
    # STEP 2: CONVERT TO GRAYSCALE
    # ========================================
    print("\n[2] Converting to grayscale for better matching...")
    
    source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    
    # Get template dimensions
    h, w = template_gray.shape
    print(f"✓ Template size: {w}x{h} pixels")
    
    # ========================================
    # STEP 3: PERFORM TEMPLATE MATCHING
    # ========================================
    print("\n[3] Performing template matching...")
    print("   Using TM_CCOEFF_NORMED method (best for general use)")
    
    # Perform template matching using correlation coefficient
    result = cv2.matchTemplate(source_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    print(f"✓ Matching complete! Result matrix size: {result.shape}")
    
    # ========================================
    # STEP 4: FIND BEST MATCH LOCATION
    # ========================================
    print("\n[4] Finding best match location...")
    
    # Find the location with maximum correlation
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    print(f"   Best match score: {max_val:.4f} (closer to 1.0 = better match)")
    print(f"   Match location: {max_loc}")
    
    # Get the top-left corner of the match
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    print(f"   Bounding box: Top-Left={top_left}, Bottom-Right={bottom_right}")
    
    # ========================================
    # STEP 5: DRAW RECTANGLE AROUND MATCH
    # ========================================
    print("\n[5] Drawing detection rectangle...")
    
    # Create a copy for drawing
    result_image = source_image.copy()
    
    # Draw rectangle (Green color, thickness=3)
    cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 3)
    
    # Add confidence text
    confidence_text = f"Match: {max_val:.2%}"
    cv2.putText(result_image, confidence_text, 
                (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    print("✓ Rectangle drawn successfully!")
    
    # ========================================
    # STEP 6: VISUALIZE RESULTS
    # ========================================
    print("\n[6] Creating visualization...")
    
    # Convert BGR to RGB for matplotlib
    source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    template_rgb = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Template Matching Results', fontsize=16, fontweight='bold')
    
    # Original Image
    axes[0, 0].imshow(source_rgb)
    axes[0, 0].set_title('1. Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Template
    axes[0, 1].imshow(template_rgb)
    axes[0, 1].set_title('2. Template to Find', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Matching Result Heatmap
    axes[1, 0].imshow(result, cmap='hot')
    axes[1, 0].set_title('3. Matching Heatmap\n(Brighter = Better Match)', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].plot(max_loc[0], max_loc[1], 'b*', markersize=20, 
                    markeredgecolor='cyan', markeredgewidth=2)
    axes[1, 0].axis('off')
    
    # Final Result
    axes[1, 1].imshow(result_rgb)
    axes[1, 1].set_title(f'4. Detection Result\nConfidence: {max_val:.2%}', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('template_matching_result.png', dpi=150, bbox_inches='tight')
    print("✓ Result saved as 'template_matching_result.png'")
    plt.show()
    
    # ========================================
    # STEP 7: TRY ALL METHODS (COMPARISON)
    # ========================================
    print("\n[7] Comparing all matching methods...")
    
    methods = {
        'TM_CCOEFF': cv2.TM_CCOEFF,
        'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
        'TM_CCORR': cv2.TM_CCORR,
        'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
        'TM_SQDIFF': cv2.TM_SQDIFF,
        'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparison of All Matching Methods', fontsize=16, fontweight='bold')
    axes = axes.ravel()
    
    for idx, (method_name, method) in enumerate(methods.items()):
        # Perform matching
        result = cv2.matchTemplate(source_gray, template_gray, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # For SQDIFF methods, minimum is best
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            match_score = min_val
            score_type = "Min"
        else:
            top_left = max_loc
            match_score = max_val
            score_type = "Max"
        
        # Draw rectangle
        img_copy = source_image.copy()
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img_copy, top_left, bottom_right, (0, 255, 0), 3)
        
        # Display
        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f'{method_name}\n{score_type}: {match_score:.4f}', 
                           fontsize=10)
        axes[idx].axis('off')
        
        print(f"   {method_name}: {score_type}={match_score:.4f}")
    
    plt.tight_layout()
    plt.savefig('all_methods_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison saved as 'all_methods_comparison.png'")
    plt.show()
    
    print("\n" + "=" * 60)
    print("TEMPLATE MATCHING COMPLETE!")
    print("=" * 60)
    print(f"\n🎯 Best Match Found at: {max_loc}")
    print(f"📊 Confidence Score: {max_val:.2%}")
    print(f"📁 Results saved in current directory")
    print("\n✨ Try adjusting threshold for multiple matches!")


def find_multiple_matches(threshold=0.8):
    """
    Advanced: Find multiple instances of the template
    """
    
    print("\n" + "=" * 60)
    print("FINDING MULTIPLE MATCHES")
    print("=" * 60)
    
    # Load images
    source_image = cv2.imread('image1.jpg')
    template_image = cv2.imread('image2.jpg')
    
    if source_image is None or template_image is None:
        print("❌ Error loading images!")
        return
    
    # Convert to grayscale
    source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape
    
    # Perform matching
    result = cv2.matchTemplate(source_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # Find all matches above threshold
    locations = np.where(result >= threshold)
    
    print(f"\n🔍 Threshold: {threshold}")
    print(f"✓ Found {len(locations[0])} potential matches")
    
    # Draw rectangles for all matches
    result_image = source_image.copy()
    
    for pt in zip(*locations[::-1]):
        cv2.rectangle(result_image, pt, (pt[0] + w, pt[1] + h), 
                     (0, 255, 0), 2)
    
    # Display
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(result_rgb)
    plt.title(f'Multiple Matches (Threshold: {threshold})', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.savefig('multiple_matches.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Results saved as 'multiple_matches.png'")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n🎯 OpenCV Template Matching Demo")
    print("📸 Cricket Players Face Detection\n")
    
    # Run the main demo
    template_matching_demo()
    
    # Optional: Try finding multiple matches
    print("\n" + "=" * 60)
    user_input = input("\nWould you like to try multiple match detection? (y/n): ")
    
    if user_input.lower() == 'y':
        threshold = float(input("Enter threshold (0.0-1.0, recommended 0.8): ") or 0.8)
        find_multiple_matches(threshold)
    
    print("\n✅ Program completed successfully!")
    print("💡 Tip: Adjust threshold values for different matching sensitivity")
    print("📚 Try with your own images by updating the file paths!\n")