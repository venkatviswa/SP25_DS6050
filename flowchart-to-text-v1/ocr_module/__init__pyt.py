import easyocr
from PIL import Image
import numpy as np
import cv2
import torch
from textblob import TextBlob
import re

# Enable GPU if available
use_gpu = torch.cuda.is_available()
reader = easyocr.Reader(['en'], gpu=use_gpu)

def expand_bbox(bbox, image_size, pad=10):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image_size[0], x2 + pad)
    y2 = min(image_size[1], y2 + pad)
    return [x1, y1, x2, y2]

def clean_text(text):
    # Basic cleanup
    text = re.sub(r'[^A-Za-z0-9?,.:;()\'"\s-]', '', text)  # remove noise characters
    text = re.sub(r'\s+', ' ', text).strip()

    # De-duplicate repeated words
    words = text.split()
    deduped = [words[0]] + [w for i, w in enumerate(words[1:], 1) if w.lower() != words[i - 1].lower()] if words else []
    joined = " ".join(deduped)

    # Run correction only if needed (long word or all caps)
    if len(joined) > 3 and any(len(w) > 10 or w.isupper() for w in deduped):
        blob = TextBlob(joined)
        joined = str(blob.correct())

    return joined

def extract_text(image, bbox, debug=False, use_adaptive_threshold=False):
    """
    Run OCR on a cropped region of the image using EasyOCR with preprocessing.

    Parameters:
        image (PIL.Image): The full image.
        bbox (list): [x1, y1, x2, y2] coordinates of the region to crop.
        debug (bool): If True, show intermediate debug output.
        use_adaptive_threshold (bool): Use adaptive thresholding instead of Otsu's.

    Returns:
        str: Extracted and cleaned text.
    """
    # Expand bbox slightly
    bbox = expand_bbox(bbox, image.size, pad=10)
    x1, y1, x2, y2 = bbox
    cropped = image.crop((x1, y1, x2, y2))

    # Convert to OpenCV format (numpy array)
    cv_img = np.array(cropped)

    # Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Resize (upscale) image for better OCR accuracy
    scale_factor = 2.5
    resized = cv2.resize(blurred, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Convert to RGB as EasyOCR expects color image
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    # Optional: debug save
    if debug:
        debug_image = Image.fromarray(resized_rgb)
        debug_image.save(f"debug_ocr_crop_{x1}_{y1}.png")

    # Run OCR using EasyOCR
    try:
        results = reader.readtext(resized_rgb, paragraph=False, min_size=5)
    except Exception as e:
        if debug:
            print(f"⚠️ EasyOCR failed: {e}")
        return ""

    if debug:
        for res in results:
            print(f"OCR: {res[1]} (conf: {res[2]:.2f})")

    # Sort boxes top to bottom, then left to right
    results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

    # Filter by confidence
    filtered = [r for r in results if r[2] > 0.4]
    if not filtered and results:
        filtered = sorted(results, key=lambda r: -r[2])[:2]  # fallback to top-2

    lines = []
    for res in filtered:
        lines.append(res[1])

    joined_text = " ".join(lines).strip()

    # Apply correction
    if joined_text:
        joined_text = clean_text(joined_text)
        if debug:
            print(f"🧹 Cleaned OCR text: {joined_text}")

    return joined_text

def count_elements(boxes, arrows, debug=False):
    box_count = len(boxes)
    arrow_count = len(arrows)
    if debug:
        print(f"📦 Detected {box_count} boxes")
        print(f"➡️  Detected {arrow_count} arrows")
    return {
        "box_count": box_count,
        "arrow_count": arrow_count
    }

def validate_structure(flowchart_json, expected_boxes=None, expected_arrows=None, debug=False):
    actual_boxes = len(flowchart_json.get("steps", []))
    actual_arrows = len(flowchart_json.get("edges", [])) if "edges" in flowchart_json else None

    if debug:
        print(f"🔍 Flowchart JSON has {actual_boxes} steps")
        if actual_arrows is not None:
            print(f"🔍 Flowchart JSON has {actual_arrows} edges")

    result = {
        "boxes_valid": (expected_boxes is None or expected_boxes == actual_boxes),
        "arrows_valid": (expected_arrows is None or expected_arrows == actual_arrows)
    }
    return result