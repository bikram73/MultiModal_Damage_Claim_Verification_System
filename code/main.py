import os
import csv
import sys

# Load .env from repo root if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
except ImportError:
    pass

from verification_engine import VerificationEngine

def run_predictions(input_csv_path, output_csv_path, strategy='heuristic'):
    if not os.path.exists(input_csv_path):
        print(f"Error: Input file {input_csv_path} does not exist.")
        return False
        
    engine = VerificationEngine()
    
    rows_to_write = []
    fieldnames = [
        'user_id', 'image_paths', 'user_claim', 'claim_object',
        'evidence_standard_met', 'evidence_standard_met_reason', 'risk_flags',
        'issue_type', 'object_part', 'claim_status', 'claim_status_justification',
        'supporting_image_ids', 'valid_image', 'severity'
    ]
    
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = engine.predict_row(row, strategy=strategy)
            rows_to_write.append(pred)
            
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_to_write:
            # Maintain exact fieldnames in order
            writer.writerow({k: row[k] for k in fieldnames})
            
    print(f"Successfully processed {len(rows_to_write)} claims. Output written to {output_csv_path}.")
    return True

if __name__ == '__main__':
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(workspace_root, 'dataset', 'claims.csv')
    output_path = os.path.join(workspace_root, 'dataset', 'output.csv')
    
    strategy = 'heuristic'
    if len(sys.argv) > 1:
        strategy = sys.argv[1]
        
    run_predictions(input_path, output_path, strategy=strategy)
