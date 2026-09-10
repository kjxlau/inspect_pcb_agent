# src/tools/ipc_standards.py
import json

with open("data/ipc_standards/ipc_a_610_chip_components.json") as f:
    IPC_DB = json.load(f)

def get_ipc_standard(component_family: str, defect_type: str, ipc_class: str = "Class_2"):
    """
    Retrieves official IPC-A-610 acceptance criteria and rework guidelines.
    """
    if "Side_Overhang" in defect_type or defect_type == "Shift":
        rule = IPC_DB["criteria"]["Side_Overhang_A"].get(ipc_class, "Standard not found")
        disposition = IPC_DB["disposition_guide"]
        return {
            "standard": "IPC-A-610 Section 9.3.1",
            "class": ipc_class,
            "threshold": rule,
            "recommended_action": disposition
        }
    return {"message": "Criteria not indexed"}
