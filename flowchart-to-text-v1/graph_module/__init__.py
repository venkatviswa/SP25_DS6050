# flowchart_builder.py
# Arrow and graph logic for converting detected flowchart elements to structured JSON

from shapely.geometry import box, Point
from collections import defaultdict, deque

def map_arrows(nodes, arrows):
    """
    Matches arrows to nodes based on geometric endpoints.
    Returns a list of (source_id, target_id, label) edges.
    """
    for node in nodes:
        node["shape"] = box(*node["bbox"])

    edges = []
    for arrow in arrows:
        tail_point = Point(arrow["tail"])
        head_point = Point(arrow["head"])
        label = arrow.get("label", "")

        source = next((n["id"] for n in nodes if n["shape"].contains(tail_point)), None)
        target = next((n["id"] for n in nodes if n["shape"].contains(head_point)), None)

        if source and target and source != target:
            edges.append((source, target, label))

    return edges

def detect_node_type(text):
    """
    Heuristic-based type detection from node text.
    """
    text_lower = text.lower()
    if "start" in text_lower:
        return "start"
    if "end" in text_lower or "full" in text_lower:
        return "end"
    if "?" in text or "yes" in text_lower or "no" in text_lower:
        return "decision"
    return "process"

def build_flowchart_json(nodes, edges):
    """
    Constructs flowchart JSON structure with parent and branching info.
    """
    graph = {}
    reverse_links = defaultdict(list)
    edge_labels = {}

    for node in nodes:
        text = node.get("text", "").strip()
        graph[node["id"]] = {
            "text": text,
            "type": node.get("type") or detect_node_type(text),
            "next": []
        }

    for src, tgt, label in edges:
        graph[src]["next"].append(tgt)
        reverse_links[tgt].append(src)
        edge_labels[(src, tgt)] = label.lower().strip()

    start_nodes = [nid for nid in graph if len(reverse_links[nid]) == 0]
    flowchart_json = {
        "start": start_nodes[0] if start_nodes else None,
        "steps": []
    }

    visited = set()
    queue = deque(start_nodes)

    while queue:
        curr = queue.popleft()
        if curr in visited:
            continue
        visited.add(curr)

        node = graph[curr]
        step = {
            "id": curr,
            "text": node["text"],
            "type": node["type"]
        }

        parents = reverse_links[curr]
        if len(parents) == 1:
            step["parent"] = parents[0]
        elif len(parents) > 1:
            step["parents"] = parents

        next_nodes = node["next"]
        if node["type"] == "decision" and len(next_nodes) >= 2:
            step["branches"] = {}
            for tgt in next_nodes:
                label = edge_labels.get((curr, tgt), "")
                if "yes" in label:
                    step["branches"]["yes"] = tgt
                elif "no" in label:
                    step["branches"]["no"] = tgt
                else:
                    step["branches"].setdefault("unknown", []).append(tgt)
            queue.extend(next_nodes)
        elif len(next_nodes) == 1:
            step["next"] = next_nodes[0]
            queue.append(next_nodes[0])
        elif len(next_nodes) > 1:
            step["next"] = next_nodes
            queue.extend(next_nodes)

        flowchart_json["steps"].append(step)

    return flowchart_json