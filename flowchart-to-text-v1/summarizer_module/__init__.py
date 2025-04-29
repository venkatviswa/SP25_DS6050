# summarizer_module/__init__.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from device_config import get_device
import torch
import json  #fixed importing json

device = get_device()

# Model config: Use phi-2-mini (replace with phi-4-mini when available)
MODEL_ID = "microsoft/Phi-4-mini-instruct"  # Ensure it's downloaded and cached locally

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
#    load_in_8bit=True, 
    torch_dtype=torch.float16
).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)

def summarize_flowchart(flowchart_json):
    """
    Generates a human-friendly explanation from flowchart JSON.

    Args:
        flowchart_json (dict): Contains "start" node and a list of "steps".

    Returns:
        str: Bullet-style explanation with proper nesting and flow.
    """
    # 📄 Prompt optimized for flow comprehension
    prompt = (
        "You are an expert in visual reasoning and instruction generation.\n"
        "Convert the following flowchart JSON into a clear, step-by-step summary using bullets.\n"
        "- Each bullet represents a process step.\n"
        "- Use indented sub-bullets to explain decision branches (Yes/No).\n"
        "- Maintain order based on dependencies and parent-child links.\n"
        "- Avoid repeating the same step more than once.\n"
        "- Do not include JSON in the output, only human-readable text.\n"
        "\nFlowchart:\n{flowchart}\n\nBullet Explanation:"
    ).format(flowchart=json.dumps(flowchart_json, indent=2))

    result = summarizer(prompt, max_new_tokens=400, do_sample=False)[0]["generated_text"]
    # Extract the portion after the final prompt marker
    if "Bullet Explanation:" in result:
        explanation = result.split("Bullet Explanation:")[-1].strip()
    else:
        explanation = result.strip()
    return explanation
