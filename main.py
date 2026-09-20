from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set in environment or .env file.")

from agent import pcb_agent_graph, PCBInspectionState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")

DEFAULT_INPUT_DIR = r"inputs"
DEFAULT_OUTPUT_FILE = "outputs/inspection_results.json"


def parse_filename_metadata(file_path: str) -> Dict[str, str]:
    """Parses board_id, component_ref, and ground_truth from image filename."""
    stem = Path(file_path).stem
    tokens = stem.split("_")

    meta = {
        "board_id": "Unknown",
        "component_ref": "Unknown",
        "ground_truth": "unknown",
        "issue_symptom": "AOI anomaly review"
    }

    if len(tokens) >= 7:
        meta["component_ref"] = tokens[1]
        meta["board_id"] = tokens[3]
        raw_gt = tokens[6].lower()

        if "missing" in raw_gt:
            meta["ground_truth"] = "missing part"
        elif "shift" in raw_gt:
            meta["ground_truth"] = "shifted"
        elif "wrong" in raw_gt:
            meta["ground_truth"] = "wrong part"
        elif "foreign" in raw_gt or "debris" in raw_gt:
            meta["ground_truth"] = "foreign material"
        elif "tombstone" in raw_gt:
            meta["ground_truth"] = "tombstone"
        elif "solder" in raw_gt:
            meta["ground_truth"] = "solder insufficient"
        elif "golden" in raw_gt or "pass" in raw_gt:
            meta["ground_truth"] = "no defect"
        else:
            meta["ground_truth"] = raw_gt

        meta["issue_symptom"] = f"AOI flagged potential defect: {meta['ground_truth']}"
    elif len(tokens) >= 2:
        meta["board_id"] = tokens[0]
        meta["component_ref"] = tokens[1]

    return meta


def run_agent_on_image(image_path: str, meta: Optional[Dict[str, str]] = None) -> PCBInspectionState:
    """Executes the monolithic inspection graph on a single image file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    if not meta:
        meta = parse_filename_metadata(image_path)

    initial_state: PCBInspectionState = {
        "image_path": str(Path(image_path).resolve()),
        "board_id": meta.get("board_id", "Unknown"),
        "component_ref": meta.get("component_ref", "Unknown"),
        "issue_symptom": meta.get("issue_symptom", "AOI defect review"),
        "historical_context": "",
        "reference_standards": "",
        "visual_description": "",
        "visual_bounding_boxes": [],
        "measurements": {},
        "defect_location": None,
        "final_defect_category": "unknown",
        "final_diagnosis_text": "",
        "grounding_confidence": 0.0,
        "self_check_passed": False,
        "errors": []
    }

    return pcb_agent_graph.invoke(initial_state)


def print_engineering_report(state: PCBInspectionState, ground_truth: Optional[str] = None):
    """Formats and displays the engineering inspection report."""
    print("\n" + "=" * 66)
    print("      PCB MONOLITHIC EXPLAINABILITY INSPECTION REPORT")
    print("=" * 66)
    print(f"Board Assembly ID:    {state['board_id']}")
    print(f"Component Reference:  {state['component_ref']}")
    if ground_truth:
        match_flag = "MATCH" if state['final_defect_category'] == ground_truth else "MISMATCH"
        print(f"Ground Truth:         {ground_truth.upper()}")
        print(f"Final Prediction:     {state['final_defect_category'].upper()} [{match_flag}]")
    else:
        print(f"Final Prediction:     {state['final_defect_category'].upper()}")

    print(f"Confidence Score:     {state['grounding_confidence'] * 100:.1f}%")
    print(f"Grounding Self-Check: {'PASSED' if state['self_check_passed'] else 'FAILED'}")

    defect_loc = state.get("defect_location")
    if defect_loc:
        print(f"Defect Localization:  {defect_loc}")

    print("-" * 66)
    print("PHYSICAL ROOT-CAUSE & IPC COMPLIANCE EXPLANATION:")
    print(state["final_diagnosis_text"])

    if state.get("errors"):
        print("-" * 66)
        print(f"Warnings/Errors: {state['errors']}")
    print("=" * 66 + "\n")


def find_all_images(base_dir: str) -> List[str]:
    """Finds all inspection images recursively."""
    image_paths = []
    if not os.path.exists(base_dir):
        return image_paths

    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, f))
    return image_paths


def run_batch(input_dir: str, limit: int = 10, output_file: str = DEFAULT_OUTPUT_FILE):
    """Processes images in batch mode and records results to JSON."""
    all_images = find_all_images(input_dir)
    if not all_images:
        logger.error(f"No images found in {input_dir}")
        return

    batch = all_images[:limit] if limit > 0 else all_images
    logger.info(f"Starting batch analysis on {len(batch)} images...")

    results = []
    correct_count = 0

    for idx, img_path in enumerate(batch, 1):
        meta = parse_filename_metadata(img_path)
        logger.info(f"[{idx}/{len(batch)}] Inspecting {meta['component_ref']} on board {meta['board_id']}...")

        final_state = run_agent_on_image(img_path, meta)
        pred = final_state["final_defect_category"].lower()
        is_correct = (pred == meta["ground_truth"])
        if is_correct:
            correct_count += 1

        results.append({
            "file_name": Path(img_path).name,
            "board_id": final_state["board_id"],
            "component_ref": final_state["component_ref"],
            "ground_truth": meta["ground_truth"],
            "predicted_defect": pred,
            "is_correct": is_correct,
            "confidence": final_state["grounding_confidence"],
            "self_check_passed": final_state["self_check_passed"],
            "defect_location": final_state.get("defect_location"),
            "diagnosis": final_state["final_diagnosis_text"],
            "errors": final_state.get("errors", [])
        })

    accuracy = (correct_count / len(batch)) * 100 if batch else 0.0
    print("\n" + "=" * 50)
    print("BATCH EVALUATION COMPLETED")
    print(f"Accuracy: {correct_count}/{len(batch)} ({accuracy:.1f}%)")
    print(f"Results written to: {output_file}")
    print("=" * 50 + "\n")

    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Monolithic PCB Explainability Agent")
    parser.add_argument("--image", type=str, help="Path to single PCB image to inspect.")
    parser.add_argument("--batch", action="store_true", help="Run batch evaluation over inputs directory.")
    parser.add_argument("--limit", type=int, default=10, help="Number of images to process in batch mode.")
    parser.add_argument("--dir", type=str, default=DEFAULT_INPUT_DIR, help="Inputs root directory.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE, help="Output JSON results file.")

    args = parser.parse_args()

    if args.image:
        meta = parse_filename_metadata(args.image)
        state = run_agent_on_image(args.image, meta)
        print_engineering_report(state, ground_truth=meta.get("ground_truth"))
    elif args.batch:
        run_batch(input_dir=args.dir, limit=args.limit, output_file=args.output)
    else:
        images = find_all_images(args.dir)
        if images:
            print("No mode flag passed. Inspecting first detected image as a sample test:")
            meta = parse_filename_metadata(images[0])
            state = run_agent_on_image(images[0], meta)
            print_engineering_report(state, ground_truth=meta.get("ground_truth"))
        else:
            logger.warning(f"No images found in {args.dir}. Use --image <path> to specify an image.")


if __name__ == "__main__":
    main()
