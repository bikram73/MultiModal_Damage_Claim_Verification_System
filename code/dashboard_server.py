import os
import csv
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from verification_engine import VerificationEngine
import main as prediction_runner

app = FastAPI(title="ClaimAI Enterprise Suite Server")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_PATH = os.path.join(WORKSPACE_ROOT, 'code', 'index.html')

NAMES = {
    'user_001': 'Alice Vance', 'user_002': 'Robert Chen', 'user_003': 'Sarah Miller',
    'user_004': 'David Kael', 'user_005': 'Emma Watson', 'user_006': 'James Smith',
    'user_007': 'Maria Garcia', 'user_008': 'John Doe', 'user_009': 'Linda Johnson',
    'user_010': 'Michael Brown', 'user_011': 'Elizabeth Taylor', 'user_012': 'William Jones',
    'user_013': 'Jennifer Miller', 'user_014': 'Richard Davis', 'user_015': 'Patricia Martinez',
    'user_016': 'Charles Anderson', 'user_017': 'Barbara Taylor', 'user_018': 'Joseph Thomas',
    'user_019': 'Susan Hernandez', 'user_020': 'Thomas Moore', 'user_021': 'Jessica Martin',
    'user_022': 'Christopher Jackson', 'user_023': 'Sarah Martin', 'user_024': 'Karen Thompson',
    'user_025': 'Nancy White', 'user_026': 'Lisa Lopez', 'user_027': 'Betty Harris',
    'user_028': 'Margaret Clark', 'user_029': 'Sandra Lewis', 'user_030': 'Ashley Robinson',
    'user_031': 'Dorothy Walker', 'user_032': 'Kimberly Young', 'user_033': 'Emily Allen',
    'user_034': 'Donna King', 'user_035': 'Michelle Wright', 'user_036': 'Carol Scott',
    'user_037': 'Amanda Green', 'user_038': 'Melissa Baker', 'user_039': 'Deborah Adams',
    'user_040': 'Stephanie Nelson', 'user_041': 'Rebecca Hill', 'user_042': 'Laura Ramirez',
    'user_043': 'Sharon Campbell', 'user_044': 'Cynthia Mitchell', 'user_045': 'Kathleen Roberts',
    'user_046': 'Amy Carter', 'user_047': 'Shirley Phillips',
}

class RunConfig(BaseModel):
    strategy: str

def compute_risk_score(pred):
    score = 15
    severity = pred.get('severity', 'none')
    status = pred.get('claim_status', 'supported')
    risk_flags = pred.get('risk_flags', 'none')
    
    if severity == 'high':
        score += 35
    elif severity == 'medium':
        score += 20
    elif severity == 'low':
        score += 10
        
    if status == 'contradicted':
        score += 25
    elif status == 'not_enough_information':
        score += 10
        
    if 'possible_manipulation' in risk_flags:
        score += 30
    if 'text_instruction_present' in risk_flags:
        score += 25
    if 'user_history_risk' in risk_flags:
        score += 15
    if 'manual_review_required' in risk_flags:
        score += 10
        
    return min(98, max(5, score))

def load_data():
    claims_path = os.path.join(WORKSPACE_ROOT, 'dataset', 'claims.csv')
    sample_path = os.path.join(WORKSPACE_ROOT, 'dataset', 'sample_claims.csv')
    output_path = os.path.join(WORKSPACE_ROOT, 'dataset', 'output.csv')
    
    claims = []
    
    # Load outputs predictions map
    output_map = {}
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                output_map[(row['user_id'], row['user_claim'].strip())] = row
                
    # Load sample claims predictions map (labeled data)
    sample_map = {}
    if os.path.exists(sample_path):
        with open(sample_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_map[(row['user_id'], row['user_claim'].strip())] = row

    # Read claims.csv (test set)
    with open(claims_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            claim_id = f"CLM-{1000 + idx}"
            key = (row['user_id'], row['user_claim'].strip())
            
            # Merge with predictions if exists
            pred = output_map.get(key, {})
            if not pred:
                # Run dynamic heuristic verification engine on the fly if not predicted yet
                engine = VerificationEngine(WORKSPACE_ROOT)
                pred = engine.predict_row(row, strategy='heuristic')
                
            policyholder = NAMES.get(row['user_id'], f"Policyholder {row['user_id']}")
            risk_score = compute_risk_score(pred)
            
            claims.append({
                'claim_id': claim_id,
                'user_id': row['user_id'],
                'policyholder': policyholder,
                'image_paths': row['image_paths'].split(';'),
                'user_claim': row['user_claim'],
                'claim_object': row['claim_object'],
                'evidence_standard_met': pred.get('evidence_standard_met', 'true'),
                'evidence_standard_met_reason': pred.get('evidence_standard_met_reason', ''),
                'risk_flags': pred.get('risk_flags', 'none').split(';'),
                'issue_type': pred.get('issue_type', 'unknown'),
                'object_part': pred.get('object_part', 'unknown'),
                'claim_status': pred.get('claim_status', 'supported'),
                'claim_status_justification': pred.get('claim_status_justification', ''),
                'supporting_image_ids': pred.get('supporting_image_ids', 'none').split(';'),
                'valid_image': pred.get('valid_image', 'true'),
                'severity': pred.get('severity', 'medium'),
                'risk_score': risk_score,
                'source': 'test'
            })
            
    # Also load sample claims as a background review source
    with open(sample_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            claim_id = f"CLM-S{2000 + idx}"
            policyholder = NAMES.get(row['user_id'], f"Policyholder {row['user_id']}")
            risk_score = compute_risk_score(row)
            
            claims.append({
                'claim_id': claim_id,
                'user_id': row['user_id'],
                'policyholder': policyholder,
                'image_paths': row['image_paths'].split(';'),
                'user_claim': row['user_claim'],
                'claim_object': row['claim_object'],
                'evidence_standard_met': row.get('evidence_standard_met', 'true'),
                'evidence_standard_met_reason': row.get('evidence_standard_met_reason', ''),
                'risk_flags': row.get('risk_flags', 'none').split(';'),
                'issue_type': row.get('issue_type', 'unknown'),
                'object_part': row.get('object_part', 'unknown'),
                'claim_status': row.get('claim_status', 'supported'),
                'claim_status_justification': row.get('claim_status_justification', ''),
                'supporting_image_ids': row.get('supporting_image_ids', 'none').split(';'),
                'valid_image': row.get('valid_image', 'true'),
                'severity': row.get('severity', 'medium'),
                'risk_score': risk_score,
                'source': 'sample'
            })
            
    return claims

@app.get("/", response_class=HTMLResponse)
def read_root():
    if not os.path.exists(HTML_PATH):
        raise HTTPException(status_code=404, detail="Dashboard index.html not found.")
    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        return f.read()

@app.get("/api/claims")
def get_claims():
    return load_data()

@app.get("/api/metrics")
def get_metrics():
    claims = load_data()
    # Compute metrics based on predictions
    total = len(claims)
    supported = sum(1 for c in claims if c['claim_status'] == 'supported')
    contradicted = sum(1 for c in claims if c['claim_status'] == 'contradicted')
    manual_review = sum(1 for c in claims if 'manual_review_required' in c['risk_flags'] or c['claim_status'] == 'not_enough_information')
    
    return {
        'total_claims': total,
        'supported': supported,
        'contradicted': contradicted,
        'manual_review': manual_review
    }

@app.post("/api/run")
def run_model(config: RunConfig):
    input_path = os.path.join(WORKSPACE_ROOT, 'dataset', 'claims.csv')
    output_path = os.path.join(WORKSPACE_ROOT, 'dataset', 'output.csv')
    
    success = prediction_runner.run_predictions(input_path, output_path, strategy=config.strategy)
    if not success:
        raise HTTPException(status_code=500, detail="Prediction runner failed.")
        
    return {"status": "success", "message": f"Successfully executed {config.strategy} prediction engine."}

@app.get("/api/images/{image_path:path}")
def serve_image(image_path: str):
    # Sanitize and construct absolute path
    sanitized_path = image_path.replace('/', os.sep).replace('\\', os.sep)
    full_path = os.path.join(WORKSPACE_ROOT, 'dataset', sanitized_path)
    
    if not os.path.exists(full_path):
        # Return fallback placeholder if file does not exist
        return HTTPException(status_code=404, detail="Image file not found.")
        
    return FileResponse(full_path)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
