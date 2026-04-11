import json
from typing import List, Dict, Any

class LayoutParser:
    def __init__(self):
        pass
    
    def parse(self, input_data: str) -> List[Dict[str, Any]]:
        """
        Parse input string or JSON into primitive dictionary.
        Supports JSON string or plain text with bbox inference.
        """
        try:
            data = json.loads(input_data)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'content' in data:
                return [data]
        except json.JSONDecodeError:
            pass

        # Fallback: treat as raw text, assume full canvas
        return [{
            "type": "text",
            "content": input_data.strip(),
            "bbox": [0.7, 0.7, 1.0, 1.0]  # Bottom-right corner
        }]