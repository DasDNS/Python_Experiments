import cv2
import numpy as np
import matplotlib as plt

def template_matching_demo ():
    print("=" * 60)
    print("TEMPLATE MATCHING DEMONSTRATION")
    print("=" * 60)

    # =======================
    # STEP 01: LOAD IMAGES
    # =======================
    print("\n[1] Loading images...")

    #Load the source image
    source_image = cv2.imread ('coke_bottle.jpg')

    #Load the template
    template_image = cv2.imread("coke_logo.jpg")

    # Check if images are loaded successfully
    if source_image is None:
        print("Error: Could not load source image")
        return
    
    if template_image is None:
        print("Error: Could not load template image")
        return
    
    print("\nSource Image loaded: {source_image.shape}")
    print("\nTemplate Image loaded: {template_image.shape}")

    # ===============
    # STEP 2: CONVERT TO GRAYSCALE
    # ===============
    print("\n[2] Converting to grayscale for better matching...")

    source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    # get template dimensions
    h, w = template_gray.shape
    print("Template size: {w}x{h} pixels")

    # ======================
    # STEP 3: PERFORM TEMPLATE MATCHING
    # ======================
    print("\n[3] Performing template matching...")
    print("Using TM_COEFF_NORMED method (best for general use)")

    #Perform template matching using correlation coefficient
    result = cv2.matchTemplate(source_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    print(f"/ Matchin complete! Result matrix size: {result.shape}")

    # =======================
    # STEP 4: FIND THE BEST MATCH LOCATION
    # =======================
    print("\n[4] Finding best match location...")

    #Find the location with maximum correlation
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(f"Best match score: {max_val:.4f} (closer to 1.0 = better match)")
    print(f"Match location: {max_loc}")

    # Get the top left corner of the image
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    print(f"Bounding box: Top-Left={top_left}, Bottom-Right={bottom_right}")

    # ===============
    # STEP 5: DRAW RECTANGLE AROUND MATCH
    # ===============
    print("\n[5] Drawing detection rectangle...")

    # Create a copy for drawing
    result_image = source_image.copy()

    # Draw rectangle (Green color, thickness = 3)
    cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 3)

    #Add confidence text
    confidence_text = f"Match: {max_val:.2%}"
    cv2.putText(result_image, confidence_text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    print("/ Rectangle drawn successfully")

    # =========
    # STEP 6: VISUALIZE RESULTS
    # =========
    print("\n[6] Creating visulaization...")

    #Convert BGR to RBG for matplotlib
    source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    template_rgb = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    #Create figure with subplots
    fig, axes = plt.subplots (2, 2, figsize=(14,10))
    
