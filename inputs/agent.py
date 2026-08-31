import logging
import json
from typing import Any, TypedDict, Literal
from PIL import Image

# Import LangGraph components
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)

from src.models.model_registry import registry 
from src.data.qdrant_store import DefectVectorStore

# ── State Definition ────────────────────────────────────────────────────────

# UPDATED: 7 defect classes + unknown
DefectType = Literal[
    "missing part", 
    "shifted", 
    "foreign material", 
    "tombstone", 
    "solder insufficient", 
    "wrong part", 
    "no defect", 
    "unknown"
]

class PCBInspectionState(TypedDict):
    # Inputs
    image: Image.Image
    board_id: str
    component_ref: str          
    issue_symptom: str          
    
    # Tool 1: Context 
    historical_context: str
    reference_standards: str
    
    # Tool 2: Visual Evidence
    visual_bounding_boxes: list[dict]
    visual_description: str
    
    # Tool 3: Measurement Evidence
    measurements: dict[str, Any] 
    
    # Tool 4: Reasoning & Grounding
    final_defect_category: DefectType
    final_diagnosis_text: str
    grounding_confidence: float
    self_check_passed: bool
    
    errors: list[str]


# ── Prompts ─────────────────────────────────────────────────────────────────

# UPDATED: Giving LLaVA physical descriptions of the 7 classes so it knows what to look for
_VISUAL_QA_PROMPT = """
Examine this Printed Circuit Board (PCB) ROI carefully. 
You must classify the visual anomaly strictly into one of the following defect types:

1. "missing part": The component is completely absent from its designated pads.
2. "shifted": The component is present but misaligned, skewed, or off-center from its pads.
3. "foreign material": Unexpected debris, contamination, or conductive material (like a loose solder ball) nearby.
4. "tombstone": A component (often a resistor or capacitor) is partially detached and standing vertically on one end.
5. "solder insufficient": The solder joint is too thin, starved, or lacks proper wetting on the component leads.
6. "wrong part": The component placed has the wrong physical package, size, or surface markings compared to standard.
7. "no defect": The component and surrounding area appear perfectly normal and correctly soldered.

Describe the exact location and morphology, and clearly state which of the defect types is visible.
"""

# UPDATED: Enforcing the 7 valid classes for the reasoning output
_REASONING_AND_GROUNDING_PROMPT = """
You are an expert PCB Failure Analysis Engineer.
Review the gathered evidence for component {component_ref} on board {board_id}.

1. SYMPTOM: {issue_symptom}
2. HISTORICAL CONTEXT: 
{historical_context}
3. VISUAL EVIDENCE: {visual_evidence}
4. MEASUREMENT EVIDENCE: {measurement_evidence}

TASKS:
A. DIAGNOSIS CATEGORY: You MUST categorize the root cause as exactly one of: ["missing part", "shifted", "foreign material", "tombstone", "solder insufficient", "wrong part", "no defect"].
B. EXPLAIN: Provide a physical explanation of the failure mode.
C. SELF-CHECK & GROUNDING: Cross-verify the visual evidence against the measurements.
   If the visual category contradicts the physical measurements, you MUST flag it.

Output strictly as JSON with keys: 
"defect_category", "explanation", "contradictions_found", "confidence_score" (0.0 to 1.0), and "self_check_passed" (boolean).
"""

def _format_similar_cases(similar: list[dict]) -> str:
    if not similar:
        return "No visually similar historical cases found in Qdrant."
    
    lines = []
    for i, s in enumerate(similar[:3], 1):
        score = s.get("score", 0.0)
        dtype = s.get("defect_category", "unknown")
        rc = s.get("root_cause", "N/A")
        lines.append(f"CASE {i} (Visual Similarity Score: {score:.2f}):\n  - Hist. Category: {dtype}\n  - Root Cause: {rc}\n")
    return "\n".join(lines)


# ── Agent Nodes (4 Tools) ───────────────────────────────────────────────────

def tool1_context_retrieval_node(state: PCBInspectionState) -> PCBInspectionState:
    logger.info("Tool 1: Gathering Qdrant Context for %s", state["component_ref"])
    errors = state.get("errors", [])
    try:
        embedding = registry.image_encoder.encode(state["image"])
        store = DefectVectorStore()
        similar_cases = store.search_similar(
            embedding=embedding, top_k=3,
            metadata_filter={"component_ref": state["component_ref"]} 
        )
        state["historical_context"] = _format_similar_cases(similar_cases)
        std_data = registry.case_db.query(component=state["component_ref"])
        state["reference_standards"] = std_data.get("ipc_standard", "Standard not defined.")
    except Exception as exc:
        logger.exception("Context Retrieval failed.")
        errors.append(f"ContextRetrieval: {exc}")
    state["errors"] = errors
    return state

def tool2_visual_evidence_node(state: PCBInspectionState) -> PCBInspectionState:
    logger.info("Tool 2: Gathering Visual Evidence")
    errors = state.get("errors", [])
    try:
        det_result = registry.pcb_detector.detect(state["image"])
        visual_desc = registry.llava.query(state["image"], _VISUAL_QA_PROMPT, max_new_tokens=300)
        state["visual_bounding_boxes"] = det_result.get("defects", [])
        state["visual_description"] = visual_desc
    except Exception as exc:
        logger.exception("Visual Evidence failed.")
        errors.append(f"VisualEvidence: {exc}")
    state["errors"] = errors
    return state
    
def tool3_measurement_evidence_node(state: PCBInspectionState) -> PCBInspectionState:
    logger.info("Tool 3: Gathering Measurement Evidence")
    errors = state.get("errors", [])
    try:
        measurements = registry.measurement_system.get_data(state["board_id"], state["component_ref"])
        state["measurements"] = measurements
    except Exception as exc:
        logger.exception("Measurement extraction failed.")
        errors.append(f"MeasurementEvidence: {exc}")
    state["errors"] = errors
    return state

def tool4_reasoning_and_grounding_node(state: PCBInspectionState) -> PCBInspectionState:
    logger.info("Tool 4: Reasoning and Self-Check")
    errors = state.get("errors", [])
    prompt = _REASONING_AND_GROUNDING_PROMPT.format(
        component_ref=state["component_ref"],
        board_id=state["board_id"],
        issue_symptom=state["issue_symptom"],
        historical_context=state["historical_context"], 
        visual_evidence=state["visual_description"],
        measurement_evidence=json.dumps(state.get("measurements", {}))
    )
    
    try:
        response_text = registry.reasoning_llm.query(prompt, require_json=True)
        
        if response_text.startswith("```json"):
            response_text = response_text.strip("`").strip("json").strip()
            
        response_data = json.loads(response_text)
        
        extracted_category = response_data.get("defect_category", "unknown").lower()
        
        # UPDATED: Validation array includes the 7 new classes
        valid_classes = [
            "missing part", "shifted", "foreign material", 
            "tombstone", "solder insufficient", "wrong part", "no defect"
        ]
        
        if extracted_category not in valid_classes:
            logger.warning("LLM returned non-standard category: %s", extracted_category)
            extracted_category = "unknown"
            
        state["final_defect_category"] = extracted_category
        state["final_diagnosis_text"] = response_data.get("explanation", "")
        state["grounding_confidence"] = float(response_data.get("confidence_score", 0.0))
        state["self_check_passed"] = bool(response_data.get("self_check_passed", False))
    except Exception as exc:
        logger.exception("Reasoning and Grounding failed.")
        errors.append(f"ReasoningGrounding: {exc}")
        state["final_defect_category"] = "unknown"
        state["self_check_passed"] = False
        
    state["errors"] = errors
    return state


# ── LangGraph Workflow Setup ─────────────────────────────────────────────────

# 1. Initialize the StateGraph with our specific TypedDict
workflow = StateGraph(PCBInspectionState)

# 2. Add our 4 nodes to the graph
workflow.add_node("context_retrieval", tool1_context_retrieval_node)
workflow.add_node("visual_evidence", tool2_visual_evidence_node)
workflow.add_node("measurement_evidence", tool3_measurement_evidence_node)
workflow.add_node("reasoning", tool4_reasoning_and_grounding_node)

# 3. Define the edges (The execution sequence)
workflow.add_edge(START, "context_retrieval")
workflow.add_edge("context_retrieval", "visual_evidence")
workflow.add_edge("visual_evidence", "measurement_evidence")
workflow.add_edge("measurement_evidence", "reasoning")
workflow.add_edge("reasoning", END)

# 4. Compile into an executable LangChain Runnable
pcb_graph = workflow.compile()
