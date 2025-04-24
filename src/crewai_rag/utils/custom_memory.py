# utils/custom_memory.py
import os
import json

class CustomMemory:
    def __init__(self, subject: str):
        self.file_path = f"memory/{subject}.json"
        os.makedirs("memory", exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump({}, f)

    def save(self, key: str, value):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        data[key] = value
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def load(self, key: str):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        return data.get(key, None)

    def clear(self) -> bool:
        """Delete the subject-specific memory file"""
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            return True
        return False
