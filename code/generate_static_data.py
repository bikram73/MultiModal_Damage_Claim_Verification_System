import os
import json
from dashboard_server import load_data

def generate():
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = load_data()
    output_path = os.path.join(workspace_root, 'code', 'claims_data.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"Static data compiled and written to {output_path}.")

if __name__ == '__main__':
    generate()
