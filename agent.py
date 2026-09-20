from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict
from langgraph.graph import StateGraph, START, END

# Import the consolidated MCP tools directly
from src.mcp.explainability_mcp_server import (
    case_context_retrieval_tool,
    visual_evidence_tool,
    measurement_evidence_tool,
    grounding_and_self_check_tool,
)

logger = logging.getLogger("pcb_agent")

DefectType = Literal[
    "missing part",
    "shifted",
    "foreign material",
    "tombstone",
    "solder insufficient",
    "wrong part",
    "no defect",
    "unknown",
]


class PCBInspectionState(TypedDict):
    # Inputs
    image_path: str
    board_id: str
    component_ref: str
    issue_symptom: str

    # Context & Tools State
    historical_context: str
    reference_standards: str
    visual_description: str
    visual_bounding_boxes: List[Dict[str, Any]]
    measurements: Dict[str, Any]

    # Final Agent Deliverables
    defect_location: Optional[Dict[str, Any]]
    final_defect_category: DefectType
    final_diagnosis_text: str
    grounding_confidence: float
    self_check_passed: bool
    errors: List[str]


_VISUAL_QA_PROMPT = """
Inspect this PCB surface mount ROI:
1. Examine component presence, alignment, solder fillet meniscus, and surrounding pad areas.
2. Identify anomalies: missing component, tombstoning, lateral shifting/misalignment, bridging, or debris.
3. Output concise morphological description and estimated bounding box coordinates.
"""

_REASONING_PROMPT_TEMPLATE = """
INSPECTION CASE DOSSIER:
- Component: {component_ref}
- Board Assembly: {board_id}
- Initial Symptom Flag: {issue_symptom}

[EVIDENCE 1: IPC STANDARDS & HISTORICAL DEFECT PRECEDENTS (MCP)]
{standards_and_precedents}

[EVIDENCE 2: VISUAL INSPECTION (LOCAL VLM MCP)]
{visual_evidence}
Observed Regions: {bounding_boxes}

[EVIDENCE 3: 3D AOI LASER & ICT ELECTRICAL TELEMETRY (MCP)]
{measurements}

TASK:
1. Cross-correlate visual findings with the physical telemetry (laser height, overhang %, ICT resistance/capacitance).
2. Detect any contradictions (e.g. solder pad discoloration vs. open-circuit 0 uF and 0 um height).
3. Validate against IPC-A-610 Class 2/3 acceptance criteria.
4. Output your diagnosis according to the requested JSON format.
"""


# ── LangGraph Pipeline Nodes ──────────────────────────────────────────────────

def node_retrieve_context(state: PCBInspectionState) -> Dict[str, Any]:
    """Node 1: Query MCP tool for historical defects and IPC standards."""
    logger.info(
        f"[Node 1: Context] Querying historical records and IPC standards for {state['component_ref']}..."
    )
    try:
        res = case_context_retrieval_tool(
            component_ref=state["component_ref"],
            board_id=state.get("board_id", ""),
            issue_symptom=state.get("issue_symptom", "")
        )
        # Handle either raw_retrievals or fallback similar_cases
        retrievals = res.get("raw_retrievals", res.get("similar_cases", []))
        return {
            "historical_context": json.dumps(retrievals, indent=2),
            "reference_standards": res.get("ipc_standard", "")
        }
    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        return {
            "historical_context": "[]",
            "reference_standards": "Standard IPC-A-610 Class 2 rules apply.",
            "errors": state.get("errors", []) + [f"Context retrieval failed: {e}"]
        }


def node_visual_inspection(state: PCBInspectionState) -> Dict[str, Any]:
    """Node 2: Query MCP tool for VLM visual inspection."""
    logger.info(f"[Node 2: Visual] Running visual feature extraction on {state['image_path']}...")
    try:
        res = visual_evidence_tool(
            image_path=state["image_path"],
            prompt=_VISUAL_QA_PROMPT
        )
        return {
            "visual_description": res.get("visual_description", "No description provided."),
            "visual_bounding_boxes": res.get("bounding_boxes", [])
        }
    except Exception as e:
        logger.error(f"Visual tool error: {e}")
        return {
            "visual_description": "Visual inference failed.",
            "visual_bounding_boxes": [],
            "errors": state.get("errors", []) + [f"Visual tool failed: {e}"]
        }


def node_measurements(state: PCBInspectionState) -> Dict[str, Any]:
    """Node 3: Query MCP tool for 3D laser & ICT measurements."""
    logger.info(f"[Node 3: Measurements] Retrieving telemetry for {state['component_ref']}...")
    try:
        telemetry = measurement_evidence_tool(
            image_path=state["image_path"],
            component_ref=state["component_ref"],
            board_id=state.get("board_id", "")
        )
        return {"measurements": telemetry}
    except Exception as e:
        logger.error(f"Telemetry tool error: {e}")
        return {
            "measurements": {},
            "errors": state.get("errors", []) + [f"Measurement tool failed: {e}"]
        }


def node_grounding_reasoning(state: PCBInspectionState) -> Dict[str, Any]:
    """Node 4: Execute multi-modal grounding, contradiction checks, and diagnosis."""
    logger.info("[Node 4: Grounding] Executing contradiction check and final synthesis...")

    standards_and_precedents = (
        f"IPC Standards & Literature:\n{state.get('reference_standards', '')}\n\n"
        f"Precedents / Context JSON:\n{state.get('historical_context', '[]')}"
    )

    reasoning_prompt = _REASONING_PROMPT_TEMPLATE.format(
        component_ref=state["component_ref"],
        board_id=state.get("board_id", "Unknown"),
        issue_symptom=state.get("issue_symptom", "Defect inspection"),
        standards_and_precedents=standards_and_precedents,
        visual_evidence=state.get("visual_description", "N/A"),
        bounding_boxes=json.dumps(state.get("visual_bounding_boxes", [])),
        measurements=json.dumps(state.get("measurements", {}), indent=2)
    )

    try:
        res = grounding_and_self_check_tool(reasoning_prompt)
        return {
            "final_defect_category": res.get("defect_category", "unknown").lower(),
            "final_diagnosis_text": res.get("explanation", "Diagnosis generated."),
            "grounding_confidence": float(res.get("confidence_score", 0.0)),
            "self_check_passed": bool(res.get("self_check_passed", False)),
            "defect_location": res.get("defect_location")
        }
    except Exception as e:
        logger.error(f"Grounding tool error: {e}")
        return {
            "final_defect_category": "unknown",
            "final_diagnosis_text": f"Diagnosis failed due to reasoning exception: {e}",
            "grounding_confidence": 0.0,
            "self_check_passed": False,
            "errors": state.get("errors", []) + [f"Grounding failed: {e}"]
        }


# ── Compile the Graph ─────────────────────────────────────────────────────────

def build_pcb_agent():
    graph = StateGraph(PCBInspectionState)

    graph.add_node("context_retrieval", node_retrieve_context)
    graph.add_node("visual_inspection", node_visual_inspection)
    graph.add_node("measurement_retrieval", node_measurements)
    graph.add_node("grounding_reasoning", node_grounding_reasoning)

    # Monolithic pipeline flow
    graph.add_edge(START, "context_retrieval")
    graph.add_edge("context_retrieval", "visual_inspection")
    graph.add_edge("visual_inspection", "measurement_retrieval")
    graph.add_edge("measurement_retrieval", "grounding_reasoning")
    graph.add_edge("grounding_reasoning", END)

    return graph.compile()


# Export compiled graph and alias
pcb_agent_graph = build_pcb_agent()
pcb_graph = pcb_agent_graph
