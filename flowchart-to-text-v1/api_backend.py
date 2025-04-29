from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io
import json
import base64

# 💡 Import modules
from yolo_module import run_yolo
from ocr_module import extract_text, count_elements, validate_structure
from graph_module import map_arrows, build_flowchart_json 
from summarizer_module import summarize_flowchart

app = FastAPI()

# 🔓 Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with actual domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-image")
async def process_image(file: UploadFile = File(...), debug: str = Form("false")):
    debug_mode = debug.lower() == "true"
    debug_log = []

    if debug_mode:
        debug_log.append("📥 Received file upload")
    print(f"📥 File received: {file.filename}")

    # 🖼️ Load image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    if debug_mode:
        debug_log.append("✅ Image converted to RGB")
    print("✅ Image converted to RGB")

    # 📦 YOLO Detection
    boxes, arrows, vis_debug = run_yolo(image)
    if debug_mode:
        debug_log.append(f"📦 Detected {len(boxes)} boxes, {len(arrows)} arrows")

    # 🔍 OCR for each box
    for box in boxes:
        box["text"] = extract_text(image, box["bbox"], debug=debug_mode)
        print(f"🔍 OCR for {box['id']}: {box['text']}")
        if debug_mode:
            debug_log.append(f"🔍 {box['id']}: {box['text']}")

    # ➡️ Build directional edges
    edges = map_arrows(boxes, arrows)
    if debug_mode:
        debug_log.append(f"➡️ Mapped {len(edges)} directional edges")

    # 🧠 Build structured flowchart
    flowchart_json = build_flowchart_json(boxes, edges)
    print("🧠 Flowchart JSON:", json.dumps(flowchart_json, indent=2))

    # ✅ Sanity checks
    structure_info = count_elements(boxes, arrows, debug=debug_mode)
    validation = validate_structure(
        flowchart_json,
        expected_boxes=structure_info["box_count"],
        expected_arrows=len(arrows),
        debug=debug_mode
    )
    if debug_mode:
        debug_log.append(f"🧾 Validation: {validation}")

    # ✍️ Generate Summary
    summary = summarize_flowchart(flowchart_json)
    print("📝 Summary:", summary)

    # 🖼️ Encode visual debug
    yolo_vis = None
    if debug_mode and vis_debug:
        vis_io = io.BytesIO()
        vis_debug.save(vis_io, format="PNG")
        yolo_vis = base64.b64encode(vis_io.getvalue()).decode("utf-8")

    return JSONResponse({
        "flowchart": flowchart_json,
        "summary": summary,
        "yolo_vis": yolo_vis,
        "debug": "\n".join(debug_log) if debug_mode else ""
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)