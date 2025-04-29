# app.py
import streamlit as st
from PIL import Image
import io
import base64
import os

# Local modules
from yolo_module import run_yolo
from ocr_module import extract_text
from graph_module import map_arrows, build_flowchart_json
from summarizer_module import summarize_flowchart

st.set_page_config(page_title="Flowchart to English", layout="wide")
st.title("📄 Flowchart to Plain English")

# Enable debug mode
debug_mode = st.toggle("🔧 Show Debug Info", value=False)

# Upload image
uploaded_file = st.file_uploader("Upload a flowchart image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Show resized preview
    max_width = 600
    ratio = max_width / float(image.size[0])
    resized_image = image.resize((max_width, int(image.size[1] * ratio)))
    st.image(resized_image, caption="📤 Uploaded Image", use_container_width=False)

    if st.button("🔍 Analyze Flowchart"):
        progress = st.progress(0, text="Detecting boxes and arrows...")
        results, arrows, vis_debug = run_yolo(image)
        progress.progress(25, text="Running OCR...")

        debug_log = []
        debug_log.append(f"📦 Detected {len(results)} boxes")
        debug_log.append(f"➡️  Detected {len(arrows)} arrows")

        for node in results:
            node["text"] = extract_text(image, node["bbox"], debug=debug_mode)
            label = node.get("label", "box")
            text = node["text"]
            debug_log.append(f"🔖 {node['id']} | Label: {label} | Text: {text}")

        progress.progress(50, text="Mapping arrows to nodes...")
        edges = map_arrows(results, arrows)

        progress.progress(75, text="Building graph structure...")
        flowchart = build_flowchart_json(results, edges)

        progress.progress(90, text="Generating explanation...")
        summary = summarize_flowchart(flowchart)

        # Show Debug Info first
        if debug_mode:
            st.markdown("### 🧪 Debug Info")
            st.code("\n".join(debug_log), language="markdown")

            st.markdown("### 🖼️ YOLO Detected Bounding Boxes")
            st.image(vis_debug, caption="YOLO Detected Boxes", use_container_width=True)

        # Show results: JSON (left), Summary (right)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🧠 Flowchart JSON")
            st.json(flowchart)
        with col2:
            st.subheader("📝 English Summary")
            st.markdown(summary)

        progress.progress(100, text="Done!")

else:
    st.info("Upload a flowchart image to begin.")