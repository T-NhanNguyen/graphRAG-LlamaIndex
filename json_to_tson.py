from typing import Any, Union, List, Dict
import numpy as np

# --- TSON Format Constants ---
TSON_FORMAT_IDENTIFIER = "tson"
MINIMUM_ITEMS_FOR_COMPRESSION = 3  

class TypeInference:
    # Maps Python types to TSON schema type identifiers for data analysis.
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "str"
    
    @staticmethod
    def inferFromValue(value: Any) -> str:
        # Infer TSON type string from a Python value.
        if isinstance(value, bool):  # Must check bool before int (bool is a subclass of int)
            return TypeInference.BOOL
        if isinstance(value, (int, np.integer)):
            return TypeInference.INT
        if isinstance(value, (float, np.floating)):
            return TypeInference.FLOAT
        return TypeInference.STRING

def isUniformDictList(dataList: List[Any]) -> bool:
    # Check if the list contains enough dictionaries to justify TSON compression.
    if not dataList or len(dataList) < MINIMUM_ITEMS_FOR_COMPRESSION:
        return False
    return all(isinstance(item, dict) for item in dataList)

def convertToTSON(dataList: List[Dict[str, Any]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    # Convert a list of dicts to TSON tabular compression to save tokens.
    if not isUniformDictList(dataList):
        return dataList
    
    # Calculate union of all keys to handle sparse dicts accurately
    allKeysSet = set()
    for item in dataList:
        allKeysSet.update(item.keys())
    
    # Sort keys for deterministic output
    fieldOrder = sorted(list(allKeysSet))
    
    # Build schema - infer type from first available value for each key
    schema = []
    for key in fieldOrder:
        # Find first non-None value for type inference across the batch
        sampleVal = next((item[key] for item in dataList if key in item and item[key] is not None), None)
        typeString = TypeInference.inferFromValue(sampleVal)
        schema.append(f"{key}:{typeString}")
    
    # Build data rows, preserving nulls where keys are missing
    dataRows = [
        [item.get(key) for key in fieldOrder]
        for item in dataList
    ]
    
    return {
        "format": TSON_FORMAT_IDENTIFIER,
        "schema": schema,
        "data": dataRows
    }