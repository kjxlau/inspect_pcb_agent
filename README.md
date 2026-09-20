# 🔍 Monolithic PCB Defect Inspection & Explainability Agent

An industrial-grade, monolithic inspection and explainability system for Printed Circuit Boards (PCBs) built using **LangGraph** and the **Model Context Protocol (MCP)**.

The system couples **Visual Evidence** (Local LLaVA Vision-Language Model), **Physical Telemetry** (3D AOI laser height profiles, coplanarity, and In-Circuit Testing electrical measurements), **IPC-A-610 Class 2/3 Standards & Root-Cause Engineering Literature** (Local Qdrant Vector RAG), and **GPT-4o Multi-Modal Reasoning** to deliver grounded, explainable root-cause diagnoses with cross-modal contradiction detection.

---

## 🏛️ System Architecture

The inspection pipeline runs as a single, sequential LangGraph state machine. Hardware telemetry, vector retrieval, and AI models are cleanly decoupled into modular **FastMCP Tools**:

```text
 ┌───────────────────────────────────────────────────────────────────────────────────────────┐
 │                        MONOLITHIC LANGGRAPH STATE MACHINE (agent.py)                      │
 │                                                                                           │
 │   [Node 1: Context]        [Node 2: Visual]        [Node 3: Telemetry]     [Node 4: Gate] │
 │   IPC Standards &     ──►  Morphological      ──►  3D Laser Profile   ──►  Grounding &    │
 │   Defect Literature        Feature Extraction      & ICT Measurements      Contradiction  │
 │   (Qdrant RAG)             (LLaVA VLM)             (3D AOI / ICT)          Check (GPT-4o) │
 └─────────────┬──────────────────────┬───────────────────────┬──────────────────────┬───────┘
               │                      │                       │                      │
               ▼                      ▼                       ▼                      ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────┐
 │                   CONSOLIDATED MCP TOOL LAYER (src/mcp/explainability_mcp_server.py)       │
 │                                                                                           │
 │ • case_context_retrieval_tool: Semantic search across IPC-A-610 PDFs & rule specs         │
 │ • visual_evidence_tool: Runs local LLaVA VLM inference over region of interest (ROI)      │
 │ • measurement_evidence_tool: O(1) lookup of 3D AOI laser profiles & ICT electrical data   │
 │ • grounding_and_self_check_tool: Cross-correlates telemetry vs. optics via GPT-4o        │
 └───────────────────────────────────────────────────────────────────────────────────────────┘
```

### Contradiction Detection & Physical Grounding

Vision-only systems frequently fail on subtle optical artifacts (e.g., mistaking shiny solder pad oxidation for component presence, or shadow occlusion for component lift). The agent resolves these edge cases by enforcing cross-modal contradiction checks:

* **Missing Part vs. Solder Starvation:** If the optical model detects slight pad discoloration or residual solder paste, but ICT telemetry reports an **infinite resistance ($R > 10\text{ M}\Omega$) / $0.0\,\mu\text{F}$ capacitance** and laser profile height is $\approx 0\,\mu\text{m}$, the self-check flags the contradiction and classifies the issue as **`missing part`**.
* **Shifted Component:** If a component is laterally displaced, the agent calculates measured overhang against pad geometry and applies **IPC-A-610 Class 2** tolerance limits (side overhang $> 50\%$ constitutes an operational defect).
* **Tombstone:** Component detached and standing on end; confirmed by an open circuit electrical reading coupled with an anomalous elevated 3D laser height profile ($> 150\text{–}250\,\mu\text{m}$) on a single termination.

---

## 🏷️ Supported Defect Taxonomy

The agent classifies anomalies into 7 IPC-aligned categories:

| Defect Class | Visual Criteria | Physical / Telemetric Verification |
| :--- | :--- | :--- |
| **`missing part`** | Exposed solder lands, absence of package body | Open circuit ($R > 10\text{ M}\Omega$ / $0.0\,\mu\text{F}$); laser height $\approx 0\,\mu\text{m}$ |
| **`shifted`** | Component package misaligned relative to lands | Side overhang exceeds $50\%$ of component width (IPC Class 2) |
| **`foreign material`**| Extraneous particles, solder splashes, flux | Visual localization; normal or shorted electrical readings |
| **`tombstone`** | Component tilted vertically on one termination | Open circuit; unilateral laser height spike |
| **`solder insufficient`**| Poor wetting fillet, exposed copper lead edge | Fillet height $< 25\%$ of component termination height |
| **`wrong part`** | Package marking mismatch, geometry discrepancy | Measured electrical value ($R, C, L$) outside nominal tolerance |
| **`no defect`** | Wetting fillet, alignment, and body nominal | Electrical value within $\pm 5\%$ tolerance; height nominal |

---

## 📂 Directory Layout

```text
Explainability_Review_Agent/
├── .env                                       # Environment variables (OPENAI_API_KEY)
├── requirements.txt                           # Project dependencies
├── inputs/                                    # Raw PCB inspection dataset
│   └── 06-200036-02/                          # Board Assembly ID
│       └── Body/
│           ├── Passed/                        # Defect image instances (MissingPart, Shift, etc.)
│           └── Golden/                        # Paired defect-free reference images
├── outputs/                                   # Telemetry datasets & evaluation logs
│   ├── synthetic_telemetry.json               # Full physical & electrical metrics
│   ├── telemetry_by_image.json                # O(1) indexed lookup dictionary
│   └── inspection_results.json                # Final diagnostic output logs
├── generate_telemetry.py                      # Synthesizes 3D AOI & ICT telemetry
├── agent.py                                   # Monolithic LangGraph agent state machine
├── main.py                                    # Unified CLI for single-image and batch runs
└── src/
    ├── mcp/
    │   └── explainability_mcp_server.py       # Consolidated FastMCP tool definitions
    ├── models/
    │   └── model_registry.py                  # Local LLaVA & OpenAI model wrappers
    └── data/
        ├── ipc_standards/                     # Source IPC PDFs, manuals, & rule specifications
        │   ├── Automated Optical Inspection_ How AOI Detects Defects.pdf
        │   ├── IPC-9712-toc.pdf
        │   ├── IPC-9716_TOC.pdf
        │   ├── IPC-A-610F.pdf
        │   ├── ipc_a_610_chip_components.json
        │   └── PCB Tombstoning_ Causes, Prevention & Fixes [2026 Updated].pdf
        ├── qdrant_db/                         # Persistent local Qdrant vector database
        ├── populate_qdrant.py                 # Ingestion & vector indexing pipeline
        ├── qdrant_store.py                    # Semantic search & retrieval interface
        └── test_query.py                      # RAG query verification test script
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
* **Python 3.10+**
* [Ollama](https://ollama.com/) (running locally)
* An **OpenAI API Key**

### 2. Pull the Local Vision Model
Ensure Ollama is running, then pull the LLaVA vision model:
```bash
ollama pull llava
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create or verify your `.env` file in the project root:
```env
OPENAI_API_KEY=sk-proj-yourActualOpenAIKeyHere
```

---

## ⚡ Execution Workflow

### Step 1: Synthesize Physical & Electrical Telemetry
Build the physical 3D AOI (laser height/overhang) and ICT (resistance/capacitance) telemetry dataset from the raw images:

```bash
python generate_telemetry.py
```
*Outputs generated:*
* `outputs/synthetic_telemetry.json` (Full telemetry dataset)
* `outputs/telemetry_by_image.json` (Indexed dictionary for $O(1)$ fast lookups)

---

### Step 2: Ingest IPC Standards & Defect Literature into Qdrant RAG
Parse, chunk, and index all documents located in `src/data/ipc_standards/` into the local vector database:

```bash
python src/data/populate_qdrant.py
```

Verify that semantic retrieval is operating correctly:
```bash
python src/data/test_query.py
```

---

### Step 3: Run the Inspection Agent

No multi-terminal orchestration or background server setup is required.

#### Option A: Inspect a Single PCB Image
```bash
python main.py --image "inputs/06-200036-02/Body/Passed/Board1_C636_Body_06-200036-02_20260824_193317036_MissingPart_3.jpg"
```

#### Option B: Run Batch Evaluation Across the Dataset
Evaluate accuracy across multiple images and write a structured summary to `outputs/inspection_results.json`:
```bash
python main.py --batch --limit 20
```

---

## 📊 Sample Output Data

### 1. Terminal Engineering Report
```text
==================================================================
      PCB MONOLITHIC EXPLAINABILITY INSPECTION REPORT
==================================================================
Board Assembly ID:    06-200036-02
Component Reference:  C636
Ground Truth:         MISSING PART
Final Prediction:     MISSING PART [MATCH]
Confidence Score:     98.0%
Grounding Self-Check: PASSED
Defect Localization:  {'landmark': 'C636 land pattern', 'bounding_box': [120, 140, 260, 310]}
------------------------------------------------------------------
PHYSICAL ROOT-CAUSE & IPC COMPLIANCE EXPLANATION:
Component C636 is entirely absent from its target land pattern. 
Physical telemetry reveals an open circuit with zero capacitance 
(0.0 uF) and near-zero laser profile height (0.8 um vs expected 45 um). 
Visual absence aligns with electrical measurements, passing the self-check.
Cited IPC Spec: IPC-A-610 Section 8.3 (Discrete Chip Components - Absence).
==================================================================
```

### 2. Final Diagnostic Output (`outputs/inspection_results.json`)
```json
[
  {
    "file_name": "Board1_C636_Body_06-200036-02_20260824_193317036_MissingPart_3.jpg",
    "board_id": "06-200036-02",
    "component_ref": "C636",
    "ground_truth": "missing part",
    "predicted_defect": "missing part",
    "is_correct": true,
    "confidence": 0.98,
    "self_check_passed": true,
    "defect_location": {
      "landmark": "C636 land pattern",
      "bounding_box": [120, 140, 260, 310]
    },
    "diagnosis": "Component C636 is entirely absent from its target land pattern. Physical telemetry reveals an open circuit with zero capacitance (0.0 uF) and near-zero laser profile height (0.8 um vs expected 45 um). Visual absence aligns with electrical measurements, passing the self-check.",
    "errors": []
  }
]
```

---

## 🛠️ Production Extensibility

* **SECS/GEM & SMT Line Integration:** In `src/mcp/explainability_mcp_server.py`, replace `measurement_evidence_tool` mock lookups with industrial SECS/GEM or REST adapters to fetch real-time telemetry from physical AOI (e.g. Koh Young, CyberOptics) and ICT fixtures (Keysight 3070, Teradyne TestStation).
* **Continuous Defect Ingestion:** When quality engineers investigate a novel manufacturing defect or validate an escalated failure (`self_check_passed == False`), add the corresponding RCA report to `src/data/ipc_standards/` and re-run `populate_qdrant.py` to immediately expand the agent's diagnostic memory.
