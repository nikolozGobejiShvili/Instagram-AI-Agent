from __future__ import annotations

import copy
import json
from pathlib import Path
from uuid import uuid4

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ENV_PATH = REPO_ROOT / "backend" / ".env"
INGESTION_COMPONENT_PATH = REPO_ROOT / "backend" / "langflow_components" / "reels_knowledge_ingestion_component.py"
GENERATION_COMPONENT_PATH = REPO_ROOT / "backend" / "langflow_components" / "reels_rag_generation_component.py"
FLOW_NAMES = {
    "ingestion": "Reels System Knowledge Ingestion",
    "generation": "Reels System RAG Generation",
}


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in BACKEND_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


class LangflowProvisioner:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = httpx.Client(
            timeout=60.0,
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self.client.close()

    def get_basic_examples(self) -> list[dict]:
        response = self.client.get(f"{self.base_url}/v1/flows/basic_examples/")
        response.raise_for_status()
        return response.json()

    def list_flows(self) -> list[dict]:
        response = self.client.get(f"{self.base_url}/v1/flows/")
        response.raise_for_status()
        return response.json()

    def delete_flow(self, flow_id: str) -> None:
        response = self.client.delete(f"{self.base_url}/v1/flows/{flow_id}")
        if response.status_code not in {200, 204}:
            response.raise_for_status()

    def register_custom_component(self, component_path: Path) -> dict:
        code = component_path.read_text(encoding="utf-8")
        response = self.client.post(
            f"{self.base_url}/v1/custom_component",
            json={"code": code},
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected custom component response for {component_path.name}")
        return {
            "type": payload.get("type"),
            "node": data,
        }

    def create_flow(self, *, name: str, description: str, data: dict) -> dict:
        response = self.client.post(
            f"{self.base_url}/v1/flows/",
            json={
                "name": name,
                "description": description,
                "data": data,
            },
        )
        response.raise_for_status()
        return response.json()


def clone_example_node(node: dict, *, node_id: str, position_x: float, position_y: float) -> dict:
    cloned = copy.deepcopy(node)
    cloned["id"] = node_id
    cloned["position"] = {"x": position_x, "y": position_y}
    cloned["type"] = "genericNode"
    if "data" in cloned and isinstance(cloned["data"], dict):
        cloned["data"]["id"] = node_id
        if isinstance(cloned["data"].get("node"), dict):
            cloned["data"]["node"]["id"] = node_id
            if isinstance(cloned["data"]["node"].get("template"), dict):
                template = cloned["data"]["node"]["template"]
                if "session_id" in template and isinstance(template["session_id"], dict):
                    template["session_id"]["value"] = ""
                if "should_store_message" in template and isinstance(template["should_store_message"], dict):
                    template["should_store_message"]["value"] = False
    return cloned


def build_custom_component_node(component_definition: dict, *, node_id: str, position_x: float, position_y: float) -> dict:
    component_node = copy.deepcopy(component_definition["node"])
    component_node["id"] = node_id
    return {
        "data": {
            "id": node_id,
            "node": component_node,
            "type": component_definition["type"],
        },
        "dragging": False,
        "id": node_id,
        "position": {"x": position_x, "y": position_y},
        "selected": False,
        "type": "genericNode",
    }


def build_edge(
    *,
    source: str,
    source_type: str,
    source_name: str,
    target: str,
    field_name: str,
    target_input_types: list[str],
) -> dict:
    source_handle = {
        "dataType": source_type,
        "id": source,
        "name": source_name,
        "output_types": ["Message"],
    }
    target_handle = {
        "fieldName": field_name,
        "id": target,
        "inputTypes": target_input_types,
        "type": "str",
    }
    return {
        "animated": False,
        "className": "",
        "data": {
            "sourceHandle": source_handle,
            "targetHandle": target_handle,
        },
        "id": f"reactflow__edge-{source}-{source_name}-{target}-{field_name}",
        "selected": False,
        "source": source,
        "sourceHandle": json.dumps(source_handle, ensure_ascii=False),
        "target": target,
        "targetHandle": json.dumps(target_handle, ensure_ascii=False),
    }


def extract_reference_nodes(basic_examples: list[dict]) -> tuple[dict, dict]:
    example = next(item for item in basic_examples if item.get("name") == "Knowledge Retrieval")
    nodes = example["data"]["nodes"]
    text_input = next(node for node in nodes if str(node.get("id", "")).startswith("TextInput"))
    chat_output = next(node for node in nodes if str(node.get("id", "")).startswith("ChatOutput"))
    return text_input, chat_output


def build_flow_graph(component_definition: dict, *, text_template: dict, output_template: dict) -> dict:
    text_node_id = f"TextInput-{uuid4().hex[:5]}"
    component_node_id = f"{component_definition['type']}-{uuid4().hex[:5]}"
    output_node_id = f"ChatOutput-{uuid4().hex[:5]}"

    text_node = clone_example_node(text_template, node_id=text_node_id, position_x=-60.0, position_y=-60.0)
    output_node = clone_example_node(output_template, node_id=output_node_id, position_x=760.0, position_y=-60.0)
    component_node = build_custom_component_node(
        component_definition,
        node_id=component_node_id,
        position_x=340.0,
        position_y=-60.0,
    )

    return {
        "nodes": [text_node, component_node, output_node],
        "edges": [
            build_edge(
                source=text_node_id,
                source_type="TextInput",
                source_name="text",
                target=component_node_id,
                field_name="payload_json",
                target_input_types=["Message"],
            ),
            build_edge(
                source=component_node_id,
                source_type=component_definition["type"],
                source_name="message",
                target=output_node_id,
                field_name="input_value",
                target_input_types=["Data", "DataFrame", "Message"],
            ),
        ],
        "viewport": {"x": 0, "y": 0, "zoom": 1},
    }


def main() -> None:
    env = load_env()
    base_url = env.get("LANGFLOW_BASE_URL", "http://127.0.0.1:7860/api")
    api_key = env.get("LANGFLOW_API_KEY", "")
    if not api_key:
        raise RuntimeError("LANGFLOW_API_KEY is missing")

    provisioner = LangflowProvisioner(base_url, api_key)
    try:
        basic_examples = provisioner.get_basic_examples()
        text_template, output_template = extract_reference_nodes(basic_examples)

        existing_flows = provisioner.list_flows()
        for flow_name in FLOW_NAMES.values():
            for flow in existing_flows:
                if flow.get("name") == flow_name and flow.get("id"):
                    provisioner.delete_flow(flow["id"])

        ingestion_component = provisioner.register_custom_component(INGESTION_COMPONENT_PATH)
        generation_component = provisioner.register_custom_component(GENERATION_COMPONENT_PATH)

        ingestion_flow = provisioner.create_flow(
            name=FLOW_NAMES["ingestion"],
            description="Internal reels knowledge ingestion into hidden Chroma vector storage.",
            data=build_flow_graph(
                ingestion_component,
                text_template=text_template,
                output_template=output_template,
            ),
        )
        generation_flow = provisioner.create_flow(
            name=FLOW_NAMES["generation"],
            description="Internal reels retrieval + generation flow using sanitized backend payloads only.",
            data=build_flow_graph(
                generation_component,
                text_template=text_template,
                output_template=output_template,
            ),
        )

        print(json.dumps({
            "ingestion_flow_id": ingestion_flow.get("id"),
            "generation_flow_id": generation_flow.get("id"),
            "vector_store_provider": env.get("LANGFLOW_VECTOR_STORE_PROVIDER", "chroma") or "chroma",
            "ingestion_flow_name": FLOW_NAMES["ingestion"],
            "generation_flow_name": FLOW_NAMES["generation"],
        }, ensure_ascii=False, indent=2))
    finally:
        provisioner.close()


if __name__ == "__main__":
    main()
