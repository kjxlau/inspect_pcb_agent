import os
import sys
import json
import logging
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

# 1. Environment & API Key Setup (supports both .env and Colab Secrets)
from dotenv import load_dotenv
load_dotenv()

# Check Google Colab userdata secrets if not in environment
try:
    from google.colab import userdata
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
except (ImportError, Exception):
    pass

if not os.environ.get("OPENAI_API_KEY"):
    print("CRITICAL WARNING: OPENAI_API_KEY is not set in environment, .env, or Colab Secrets!")

# 2. Import compiled LangGraph workflow & State
from agent import pcb_graph, PCBInspectionState

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Auto-detect Colab paths vs standard relative paths
COLAB_BASE_DIR = Path("/content/inspect_pcb_agent/inputs")
DEFAULT_INPUT_DIR = str(COLAB_BASE_DIR) if COLAB_BASE_DIR.exists() else "./inputs"
DEFAULT_OUTPUT_FILE = "inspection_results.json"


# ── Colab / Ollama Pre-flight Check ──────────────────────────────────────────

def check_ollama_service() -> bool:
    """Verifies that Ollama server is running locally (required for LLaVA node)."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


# ── Metadata Parser ──────────────────────────────────────────────────────────

def parse_filename_metadata(file_path: str) -> Dict[str, str]:
    """
    Extracts metadata from filenames like:
    'Board1_C978_Body_06-200036-02_20260824_154737922_WrongPart_13.jpg'
    """
    stem = Path(file_path).stem
    tokens = stem.split("_")

    meta = {
        "board_id": "Unknown",
        "component_ref": "Unknown",
        "ground_truth": "unknown",
        "issue_symptom": "AOI flagged anomaly"
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


# ── Core Agent Invocation ───────────────────────────────────────────────────

def run_agent_on_image(image_path: str, meta: Optional[Dict[str, str]] = None) -> PCBInspectionState:
    """Executes the full LangGraph agent workflow on a given image."""
    abs_path = str(Path(image_path).resolve())
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Image not found at {abs_path}")

    # Ollama health check before calling graph
    if not check_ollama_service():
        logger.warning(
            "Ollama does not appear to be running on http://127.0.0.1:11434!\n"
            "If the visual inspection fails, start Ollama in Colab using:\n"
            "  !curl -fsSL https://ollama.com/install.sh | sh\n"
            "  !nohup ollama serve > ollama.log 2>&1 &\n"
            "  !ollama pull llava\n"
        )

    if not meta:
        meta = parse_filename_metadata(abs_path)

    # Note: image_path is explicitly included here for node2 / visual tool
    initial_state: PCBInspectionState = {
        "image_path": abs_path,
        "image": Image.open(abs_path).convert("RGB"),
        "board_id": meta.get("board_id", "Unknown"),
        "component_ref": meta.get("component_ref", "Unknown"),
        "issue_symptom": meta.get("issue_symptom", "AOI anomaly review"),
        "historical_context": "",
        "reference_standards": "",
        "visual_bounding_boxes": [],
        "visual_description": "",
        "measurements": {},
        "defect_location": None,
        "final_defect_category": "unknown",
        "final_diagnosis_text": "",
        "grounding_confidence": 0.0,
        "self_check_passed": False,
        "errors": []
    }

    return pcb_graph.invoke(initial_state)


# ── Display Formatter ────────────────────────────────────────────────────────

def print_engineering_report(state: PCBInspectionState, ground_truth: Optional[str] = None):
    """Prints a structured engineering report for human review."""
    print("\n" + "=" * 60)
    print("        PCB DEFECT EXPLAINABILITY REVIEW REPORT")
    print("=" * 60)
    print(f"Board Assembly ID:    {state.get('board_id', 'Unknown')}")
    print(f"Component Reference:  {state.get('component_ref', 'Unknown')}")
    final_pred = state.get('final_defect_category', 'unknown')
    if ground_truth:
        match_icon = "MATCH" if final_pred == ground_truth else "MISMATCH"
        print(f"Ground Truth:         {ground_truth.upper()}")
        print(f"Final Prediction:     {final_pred.upper()} [{match_icon}]")
    else:
        print(f"Final Prediction:     {final_pred.upper()}")
        
    print(f"Confidence Score:     {state.get('grounding_confidence', 0.0) * 100:.1f}%")
    print(f"Grounding Self-Check: {'PASSED' if state.get('self_check_passed') else 'FAILED'}")

    defect_loc = state.get("defect_location")
    if isinstance(defect_loc, dict):
        landmark = defect_loc.get("landmark", "Unspecified")
        bbox = defect_loc.get("bounding_box")
        print(f"Defect Landmark:      {landmark}")
        if bbox:
            print(f"Bounding Box [y,x]:   {bbox}")
    elif defect_loc:
        print(f"Defect Location:      {defect_loc}")
    elif state.get("visual_bounding_boxes"):
        print(f"Bounding Box [y,x]:   {state['visual_bounding_boxes']}")
    else:
        print("Defect Location:      None / Not localized")

    print("-" * 60)
    print("PHYSICAL ROOT-CAUSE & IPC COMPLIANCE EXPLANATION:")
    print(state.get("final_diagnosis_text", ""))
    
    if state.get("errors"):
        print("-" * 60)
        print(f"Warnings/Errors: {state['errors']}")
    print("=" * 60 + "\n")


# ── Image Finder ─────────────────────────────────────────────────────────────

def find_all_images(base_dir: str) -> List[str]:
    """Recursively finds all defect images inside Passed/ subfolders."""
    image_paths = []
    base_path = Path(base_dir).resolve()
    if not base_path.exists():
        logger.error(f"Input directory does not exist: {base_path}")
        return image_paths

    for root, _, files in os.walk(str(base_path)):
        if "passed" in root.lower():
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, f))
    return image_paths


# ── Execution Modes ──────────────────────────────────────────────────────────

def run_single_image_mode(image_path: str):
    """Evaluates a single image and prints detailed diagnosis."""
    logger.info(f"Running Single-Image Review on: {image_path}")
    meta = parse_filename_metadata(image_path)
    final_state = run_agent_on_image(image_path, meta)
    print_engineering_report(final_state, ground_truth=meta.get("ground_truth"))


def run_batch_mode(input_dir: str, limit: int = 10, output_file: str = DEFAULT_OUTPUT_FILE):
    """Evaluates a batch of images and saves evaluation metrics."""
    all_images = find_all_images(input_dir)
    logger.info(f"Found {len(all_images)} total images in: {input_dir}")

    if not all_images:
        logger.error("No images found! Check your input directory path.")
        return

    test_batch = all_images[:limit] if limit > 0 else all_images
    logger.info(f"Starting batch evaluation on {len(test_batch)} images...\n")

    results = []
    correct_count = 0

    for idx, img_path in enumerate(test_batch, 1):
        filename = os.path.basename(img_path)
        meta = parse_filename_metadata(img_path)

        logger.info(f"[{idx}/{len(test_batch)}] Processing: {meta['component_ref']} on Board {meta['board_id']}")
        logger.info(f"  Ground Truth: '{meta['ground_truth']}'")

        try:
            final_state = run_agent_on_image(img_path, meta)
            predicted_cat = final_state.get("final_defect_category", "unknown").lower()
            is_correct = (predicted_cat == meta["ground_truth"])

            if is_correct:
                correct_count += 1
                logger.info(f"  ✔ Prediction: '{predicted_cat}' [MATCH]\n")
            else:
                logger.warning(f"  ✖ Prediction: '{predicted_cat}' [MISMATCH]\n")

            results.append({
                "file_name": filename,
                "file_path": img_path,
                "board_id": final_state.get("board_id"),
                "component_ref": final_state.get("component_ref"),
                "ground_truth": meta["ground_truth"],
                "predicted_defect": predicted_cat,
                "defect_location": final_state.get("defect_location"),
                "is_correct": is_correct,
                "confidence": final_state.get("grounding_confidence"),
                "self_check_passed": final_state.get("self_check_passed"),
                "diagnosis": final_state.get("final_diagnosis_text"),
                "errors": final_state.get("errors")
            })

        except Exception as exc:
            logger.error(f"Failed to process {filename}: {exc}")

    acc = (correct_count / len(test_batch)) * 100 if test_batch else 0
    print("\n" + "=" * 50)
    print("BATCH EVALUATION COMPLETE")
    print(f"Accuracy: {correct_count}/{len(test_batch)} ({acc:.1f}%)")
    print(f"Results saved to: {output_file}")
    print("=" * 50 + "\n")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


# ── Main Entrypoint ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PCB Explainability Review Agent")
    parser.add_argument("--image", type=str, help="Path to a single PCB image to inspect.")
    parser.add_argument("--batch", action="store_true", help="Run batch evaluation over images directory.")
    parser.add_argument("--limit", type=int, default=10, help="Number of images to process in batch mode (default: 10).")
    parser.add_argument("--dir", type=str, default=DEFAULT_INPUT_DIR, help="Base input folder path.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE, help="Output JSON results filename.")

    # Filter out Jupyter/IPython internal arguments when running in notebooks
    clean_argv = [a for a in sys.argv[1:] if not a.startswith("-f") and "kernel" not in a]
    args = parser.parse_args(clean_argv)

    if args.image:
        run_single_image_mode(args.image)
    elif args.batch:
        run_batch_mode(input_dir=args.dir, limit=args.limit, output_file=args.output)
    else:
        all_images = find_all_images(args.dir)
        if all_images:
            print(f"Inspecting first found image in '{args.dir}':")
            run_single_image_mode(all_images[0])
            print("\nTip: To run batch evaluation on 10 images, run:\n  !python main.py --batch --limit 10")
        else:
            logger.warning(f"No images found in {args.dir}. Specify an image with --image <path>.")


if __name__ == "__main__":
    main()
