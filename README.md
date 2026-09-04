# 🔍 AI-Powered PCB Defect Inspection Agent

This project is an autonomous multi-agent framework built with **LangGraph** to inspect Printed Circuit Boards (PCBs) for manufacturing defects. It leverages **Local Vision-Language Models (LLaVA)**, **Local LLMs (Llama 3.1)**, and a **Vector Database (Qdrant)** to perform visual inspection, retrieve historical failure contexts, and reason through physical measurements—all without sending sensitive manufacturing data to the cloud.

## ✨ Features
* **100% Local & Private:** Uses Ollama to run AI models entirely on your local machine. No intellectual property leaves the factory.
* **LangGraph Orchestration:** Utilizes a state-graph architecture to route PCB data through specialized "Agent Nodes" sequentially.
* **Vector-Based RAG (Retrieval-Augmented Generation):** Automatically searches an in-memory Qdrant database for historically similar defects to aid in root-cause analysis.
* **7-Class Defect Taxonomy:** Capable of detecting:
  1. `missing part`
  2. `shifted`
  3. `foreign material`
  4. `tombstone`
  5. `solder insufficient`
  6. `wrong part`
  7. `no defect`

---

## 📂 Project Structure

```text
Prototype Code/
│
├── inputs/                      # Folder containing raw PCB .png images
├── metadata.csv                 # (Optional) Maps images to specific components/symptoms
├── main.py                      # Main script to batch process images
├── agent.py                     # LangGraph workflow, State definition, and Prompts
│
└── src/
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   └── model_registry.py    # Ollama integrations (LLaVA & Llama)
    │
    └── data/
        ├── __init__.py
        └── qdrant_store.py      # Local In-Memory Vector DB for historical context
```

---

## ⚙️ How the Pipeline Works

The system passes a `PCBInspectionState` dictionary through 4 primary Agent Nodes via LangGraph:

1. **Tool 1: Context Retrieval (`qdrant_store.py`)** 
   Takes the target component name and finds visually similar historical defect cases in the vector database to establish precedents.
2. **Tool 2: Visual Evidence (`model_registry.py` -> `LLaVA`)** 
   Passes the image to a Local Vision Model (LLaVA) configured with a strict prompt to identify morphology and classify the visual anomaly.
3. **Tool 3: Measurement Evidence** 
   Queries backend databases (simulated) for physical/electrical measurements (e.g., In-Circuit Test resistance, 3D AOI height).
4. **Tool 4: Reasoning & Grounding (`model_registry.py` -> `Llama 3.1`)** 
   Synthesizes the visual description, historical context, and physical measurements. It cross-checks visual claims against physical data to detect contradictions and outputs a final JSON diagnosis.
```
┌─────────────────────────────────────────────────────────┐
 │                   LangGraph Workflow                    │
 │                                                         │
 │  [Tool 1: Context]  ───┐                                │
 │  [Tool 2: Visual]   ───┼──► OpenAI (GPT-4o Reasoning)  │
 │  [Tool 3: Measure]  ───┘   (Structured JSON Output)     │
 └─────────────┬───────────────────────────────────────────┘
               │ Calls via MCP Protocol
 ┌─────────────▼───────────────────────────────────────────┐
 │                    PCB MCP Server                       │
 │  • search_historical_defects (Qdrant Vector RAG)        │
 │  • get_ipc_standards (Case DB)                          │
 │  • get_ict_measurements (ICT / Laser Profile System)    │
 └─────────────────────────────────────────────────────────┘
```
---

## 🚀 Setup & Installation

### 1. Install Prerequisites
* Python 3.9+
* [Ollama](https://ollama.com/) (Installed and running on your local machine)

### 2. Download Local AI Models
Open your terminal / command prompt and pull the required models into Ollama:
```bash
ollama pull llava
```

### 3. Install Python Dependencies
Run the following command to install the required Python libraries:
```bash
pip install langgraph qdrant-client ollama pillow
```

---

## 💻 Usage

### 1. Prepare your Data
Ensure your PCB images (e.g., `100_input.png`) are located in the `inputs/` folder.
*(Optional)* Create a `metadata.csv` in the root folder to map images to specific components:
```csv
filename,component_ref,issue_symptom
100_input.png,U12,AOI flagged anomaly
101_input.png,C42,Possible short circuit
```

### 2. Run the Batch Processor
Execute the main script from your terminal:
```bash
python main.py
```

### 3. View the Results
The agent will process the images (defaulting to the first 5 images for testing purposes). Once completed, it generates an `inspection_results.json` file in the root directory.

**Example Output:**
```json
[
    {
        "board_id": "100",
        "component_ref": "U12",
        "file_name": "100_input.png",
        "defect_category": "tombstone",
        "confidence": 0.92,
        "diagnosis": "The capacitor is detached at one pad and standing vertically due to uneven reflow heating. Historical cases confirm this behavior under similar oven profiles.",
        "self_check_passed": true,
        "errors": []
    }
]
```

---

## 🛠 Configuration & Customization

* **Process All Images:** In `main.py`, change `test_files = all_files[:5]` to `test_files = all_files` to run your entire dataset.
* **Adding Real Databases:** Open `src/models/model_registry.py` and replace `MockMeasurementSystem`, `MockCaseDB`, and `MockDetector` with standard API calls to your factory's actual MES/ICT databases.
