# ClaimAI System - Code Module

This directory contains the core implementation of the **ClaimAI** Multi-Modal Evidence Review and verification system, including the FastAPI interactive dashboard.

## Folder Structure

```text
code/
├── evaluation/
│   ├── main.py                   # Evaluation script
│   └── evaluation_report.md       # Metrics & operational report
├── README.md                     # Documentation
├── main.py                       # Suggested entry point for CLI predictions
├── verification_engine.py         # Claims classifier logic & quality filters
├── dashboard_server.py           # FastAPI backend server
└── index.html                    # Dynamic UI dashboard template
```

## Running Predictions

To run batch predictions on `dataset/claims.csv` and write results to `dataset/output.csv`:

```bash
python code/main.py [strategy]
```
- Available strategies: `heuristic` (default), `vlm`

## Running Evaluation

To evaluate prediction accuracy on `dataset/sample_claims.csv` and generate the metrics report:

```bash
python code/evaluation/main.py
```

## Launching the Web Dashboard

To run the interactive FastAPI dashboard:

```bash
python code/dashboard_server.py
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

- **Metrics Cards:** Real-time updates on total claims, supported, contradicted, and manual review counts.
- **Interactive Table:** Click any row in the **Risk Analysis List** to load its details.
- **Evidence Viewer:** Renders actual images and marks issues.
- **AI Report & Justification:** Displays risk flags, estimated severity, target parts, and natural language explanations.
- **Search & Filter:** Instantly filter table contents by Claim ID, policyholder name, or object.
- **Run Verification:** Dynamically trigger prediction engine strategy changes from the UI.
