import os
import csv
import json
import logging
from PIL import Image
from typing import Dict, Any

# 1. Import the compiled LangGraph app!
from agent import pcb_graph, PCBInspectionState

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

INPUT_DIR = r"C:\Users\kenny\Desktop\Semicon Agents\Prototype Code\inputs"
METADATA_FILE = "metadata.csv"
OUTPUT_FILE = "inspection_results.json"

def process_single_image(image_path: str, board_id: str, component_ref: str, issue_symptom: str) -> Dict[str, Any]:
    """Runs the LangGraph PCB inspection pipeline on a single image."""
    
    # Initialize the starting state
    initial_state: PCBInspectionState = {
        "image": Image.open(image_path).convert("RGB"),
        "board_id": board_id,
        "component_ref": component_ref,       
        "issue_symptom": issue_symptom,       
        "historical_context": "",
        "reference_standards": "",
        "visual_bounding_boxes": [],
        "visual_description": "",
        "measurements": {},
        "final_defect_category": "unknown",
        "final_diagnosis_text": "",
        "grounding_confidence": 0.0,
        "self_check_passed": False,
        "errors": []
    }

    # 2. Execute the LangGraph pipeline
    # The invoke() method routes the state through all the nodes automatically
    logger.info(f"Triggering LangGraph for {board_id}...")
    final_state = pcb_graph.invoke(initial_state)

    # 3. Format the result to save
    return {
        "board_id": final_state["board_id"],
        "component_ref": final_state["component_ref"],
        "file_name": os.path.basename(image_path),
        "defect_category": final_state["final_defect_category"],
        "confidence": final_state["grounding_confidence"],
        "diagnosis": final_state["final_diagnosis_text"],
        "self_check_passed": final_state["self_check_passed"],
        "errors": final_state["errors"]
    }

def load_metadata() -> Dict[str, Dict[str, str]]:
    metadata = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row["filename"]] = {
                    "component_ref": row.get("component_ref", "U1"),
                    "issue_symptom": row.get("issue_symptom", "AOI anomaly")
                }
    return metadata

def main():
    metadata = load_metadata()
    results = []
    
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_input.png")]
    test_files = all_files[:5] # Change to all_files when ready

    logger.info(f"Starting LangGraph batch process for {len(test_files)} images...")

    for filename in test_files:
        image_path = os.path.join(INPUT_DIR, filename)
        board_id = filename.split('_')[0]
        
        file_meta = metadata.get(filename, {"component_ref": "U12", "issue_symptom": "AOI flagged anomaly"})
        comp_ref = file_meta["component_ref"]
        symptom = file_meta["issue_symptom"]

        try:
            result = process_single_image(image_path, board_id, comp_ref, symptom)
            results.append(result)
            logger.info(f"Result for {comp_ref}: {result['defect_category']} (Conf: {result['confidence']})")
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Batch processing complete! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
