# yolo_module.py
from ultralytics import YOLO
from device_config import get_device
from PIL import Image, ImageDraw
import numpy as np
import easyocr

# Load YOLO model
MODEL_PATH = "models/best.pt"
device = get_device()
model = YOLO(MODEL_PATH).to(device)
print(f"✅ YOLO model loaded on: {device}")


# Optional OCR reader for arrow label detection
reader = easyocr.Reader(['en'], gpu=False)

def run_yolo(image: Image.Image):
    results = model.predict(image, conf=0.25, verbose=False)[0]

    boxes = []
    arrows = []

    # Convert image to OpenCV format for EasyOCR
    np_img = np.array(image)

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = [x1, y1, x2, y2]

        item = {
            "id": f"node{i+1}",
            "bbox": bbox,
            "type": "arrow" if label in ["arrow", "control_flow"] else "box",
            "label": label
        }

        if item["type"] == "arrow":
            # Heuristically scan a small region near the middle of the arrow for a label
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            pad = 20
            crop = np_img[max(cy - pad, 0):cy + pad, max(cx - pad, 0):cx + pad]

            detected_label = ""
            if crop.size > 0:
                ocr_results = reader.readtext(crop)
                if ocr_results:
                    detected_label = ocr_results[0][1]  # (bbox, text, conf)

            arrows.append({
                "id": f"arrow{len(arrows)+1}",
                "tail": (x1, y1),
                "head": (x2, y2),
                "label": detected_label
            })
        else:
            boxes.append(item)

    vis_image = results.plot(pil=True)
    return boxes, arrows, vis_image