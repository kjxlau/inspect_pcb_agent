import os
from pathlib import Path

# Define base path
BASE = Path("/content/inspect_pcb_agent")

# 1. Create directories
(BASE / "src" / "mcp").mkdir(parents=True, exist_ok=True)
(BASE / "src" / "models").mkdir(parents=True, exist_ok=True)
(BASE / "src" / "data").mkdir(parents=True, exist_ok=True)

# 2. Ensure __init__.py files exist
for pkg_dir in [BASE / "src", BASE / "src" / "mcp", BASE / "src" / "models", BASE / "src" / "data"]:
    init_file = pkg_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        print(f"Created: {init_file}")

# 3. Write the exact explainability_mcp_server.py
server_code = '''"""Consolidated FastMCP Tool definitions for the Monolithic PCB Explainability Agent."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("pcb-explainability-tools")
logger = logging.getLogger(__name__)

# Telemetry paths and cache
TELEMETRY_INDEX_PATH = Path("outputs/telemetry_by_image.json")
_TELEMETRY_CACHE: Optional[Dict[str, Any]] = None
_OPENAI_CLIENT = None


def _load_telemetry() -> Dict[str, Any]:
    """Caches and returns synthetic AOI & ICT telemetry data."""
    global _TELEMETRY_CACHE
    if _TELEMETRY_CACHE is None:
        if TELEMETRY_INDEX_PATH.exists():
            try:
                with open(TELEMETRY_INDEX_PATH, "r", encoding="utf-8") as f:
                    _TELEMETRY_CACHE = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read telemetry cache at {TELEMETRY_INDEX_PATH}: {e}")
                _TELEMETRY_CACHE = {}
        else:
            logger.warning(f"Telemetry cache not found at {TELEMETRY_INDEX_PATH}")
            _TELEMETRY_CACHE = {}
    return _TELEMETRY_CACHE


def _get_openai_client():
    """Initializes and caches the OpenAI client."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing.")
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


# ── MCP Tool 1: Case Context & IPC Standards RAG Retrieval ────────────────────
@mcp.tool()
def case_context_retrieval_tool(
    component_ref: str,
    board_id: str = "",
    issue_symptom: str = ""
) -> Dict[str, Any]:
    """Retrieves relevant IPC-A-610 Class 2/3 inspection criteria, AOI detection
    whitepapers, and defect troubleshooting literature from local Qdrant.
    """
    logger.info(f"Context Retrieval Query: comp={component_ref}, board={board_id}, symptom={issue_symptom}")

    default_ipc_standard = (
        "IPC-A-610 Class 2/3 Acceptability Requirements:\\n"
        "- Missing Part: Component absent from designated land pattern.\\n"
        "- Shifted: Maximum allowable side overhang is <= 50% of component termination width.\\n"
        "- Tombstone: Component detached at one end; tilt angle > 0 deg with lifted terminal.\\n"
        "- Solder Insufficient: Fillet height < 25% of component termination height.\\n"
        "- Foreign Material: Debris, flux residue, or solder splatter bridging conductors."
    )

    retrieved_documents: List[Dict[str, Any]] = []

    try:
        from src.data.qdrant_store import query_defect_precedents

        query_text = (
            f"Defect symptom '{issue_symptom}' on component {component_ref} board {board_id}. "
            f"IPC-A-610 acceptance criteria, causes, AOI detection, and prevention."
        )

        retrieved_documents = query_defect_precedents(query_text=query_text, top_k=4)

    except Exception as exc:
        logger.warning(f"Qdrant retrieval fallback triggered: {exc}")

    if retrieved_documents:
        formatted_sections = []
        for i, doc in enumerate(retrieved_documents, 1):
            source = doc.get("source", "Standard Reference")
            page = f" (Page {doc['page']})" if doc.get("page") else ""
            score = doc.get("score", 0.0)
            text = doc.get("text", "").strip()
            formatted_sections.append(
                f"[Document {i}] Source: {source}{page} | Similarity: {score}\\n{text}"
            )
        retrieved_context_text = "\\n\\n".join(formatted_sections)
    else:
        retrieved_context_text = default_ipc_standard

    return {
        "ipc_standard": retrieved_context_text,
        "raw_retrievals": retrieved_documents
    }


# ── MCP Tool 2: Visual Evidence Extraction (LLaVA / VLM) ──────────────────────
@mcp.tool()
def visual_evidence_tool(image_path: str, prompt: str) -> Dict[str, Any]:
    """Inspects the PCB region of interest (ROI) using the local LLaVA vision model."""
    logger.info(f"Visual Extraction Tool invoked on: {image_path}")
    try:
        from src.models.model_registry import query_local_llava
        return query_local_llava(image_path=image_path, prompt=prompt)
    except Exception as exc:
        logger.warning(f"Local VLM inference fallback triggered: {exc}")
        return {
            "visual_description": "Component site shows target land pads with solder paste impression.",
            "bounding_boxes": [{"box_2d": [120, 140, 260, 310], "label": "component_site"}]
        }


# ── MCP Tool 3: Physical & Electrical Telemetry Lookup ─────────────────────────
@mcp.tool()
def measurement_evidence_tool(image_path: str, component_ref: str, board_id: str) -> Dict[str, Any]:
    """Fetches physical 3D AOI (laser height profile, coplanarity, side overhang)
    and ICT (In-Circuit Testing resistance/capacitance) telemetry.
    """
    telemetry_db = _load_telemetry()
    filename = Path(image_path).name

    if filename in telemetry_db:
        return telemetry_db[filename]

    for item in telemetry_db.values():
        if item.get("component_ref") == component_ref and item.get("board_id") == board_id:
            return item

    logger.info(f"Telemetry missing for {filename}. Using default nominal profile.")
    return {
        "board_id": board_id,
        "component_ref": component_ref,
        "nominal_value": 10.0,
        "measured_value": 10.0,
        "unit": "kOhm",
        "ict_status": "PASS",
        "laser_profile_height_um": 42.0,
        "side_overhang_percent": 2.0,
        "coplanarity_um": 1.2,
        "aoi_status": "PASS"
    }


# ── MCP Tool 4: Grounding & Self-Check Reasoning ──────────────────────────────
@mcp.tool()
def grounding_and_self_check_tool(reasoning_prompt: str) -> Dict[str, Any]:
    """Executes multi-modal cross-verification and contradiction checks using OpenAI GPT-4o."""
    logger.info("Executing Grounding & Self-Check Reasoning via GPT-4o...")
    client = _get_openai_client()

    system_prompt = (
        "You are an IPC-A-610 Master Review Inspector for SMT assembly quality assurance.\\n"
        "Your task is to analyze all visual descriptions, physical 3D AOI laser measurements, "
        "electrical ICT readings, and retrieved IPC standards to identify contradictions and determine "
        "the grounded root cause.\\n\\n"
        "CONTRADICTION & REASONING RULES:\\n"
        "1. Missing Part: An open circuit (R > 10 MOhm or C = 0.0 uF) and laser profile height near 0 um "
        "overrides visual discoloration or paste presence. Flag as 'missing part'.\\n"
        "2. Shifted: If component overhang > 50% of termination width, flag as 'shifted' under IPC Class 2.\\n"
        "3. Tombstone: Open circuit + one elevated terminal (> 150-250 um) signifies tombstoning.\\n"
        "4. Solder Insufficient: Fillet height < 25% of termination height.\\n"
        "5. Wrong Part: Measured electrical value outside tolerance band or package dimensions mismatch.\\n\\n"
        "Respond ONLY with a valid JSON object matching this schema:\\n"
        "{\\n"
        '  "defect_category": "missing part" | "shifted" | "foreign material" | "tombstone" | "solder insufficient" | "wrong part" | "no defect",\\n'
        '  "confidence_score": float (0.0 to 1.0),\\n'
        '  "self_check_passed": bool,\\n'
        '  "defect_location": {"landmark": string, "bounding_box": [y1, x1, y2, x2] or null},\\n'
        '  "explanation": string\\n'
        "}"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": reasoning_prompt}
        ]
    )

    return json.loads(response.choices[0].message.content)
'''

target_file = BASE / "src" / "mcp" / "explainability_mcp_server.py"
target_file.write_text(server_code, encoding="utf-8")
print(f"Successfully wrote {target_file.stat().st_size} bytes to {target_file}")
