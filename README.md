# 🛡️ ClaimAI — Multi-Modal Evidence Review

An enterprise-grade AI system that verifies physical damage claims using submitted images, conversation transcripts, user claim history, and evidence requirements.

[![GitHub](https://img.shields.io/badge/GitHub-bikram73%2FMultiModal__Damage__Claim__Verification__System-181717?logo=github)](https://github.com/bikram73/MultiModal_Damage_Claim_Verification_System)
[![Live Demo](https://img.shields.io/badge/Vercel-ClaimAI%20Dashboard-000000?logo=vercel)](https://claim-ai-multi-modal-evidence-review.vercel.app)

---

## What this project does

ClaimAI processes insurance damage claims for three object types — **car**, **laptop**, and **package** — and produces a structured verdict for each one:

- **Supported** — image evidence confirms the claimed damage
- **Contradicted** — image evidence conflicts with the claim
- **Not Enough Information** — images are insufficient to decide

For every claim the system:

1. Parses the chat transcript to extract the exact damage type, object, and part
2. Runs image quality checks (blur, low light, glare, OCR-based prompt injection)
3. Validates the image set against minimum evidence requirements
4. Cross-references the user's claim history for risk context
5. Produces a grounded justification referencing specific image IDs
6. Estimates damage severity and flags anomalies

---

## Live links

| | Link |
|---|---|
| GitHub | https://github.com/bikram73/MultiModal_Damage_Claim_Verification_System |
| Live Dashboard | https://claim-ai-multi-modal-evidence-review.vercel.app |

---

## Repository structure

```text
.
├── AGENTS.md                        # Agent onboarding rules and log registry
├── problem_statement.md             # Full task description and I/O schema
├── requirements.txt                 # Python dependencies
├── vercel.json                      # Vercel static deployment config
├── .env                             # API keys (never committed)
├── .gitignore
├── code/
│   ├── main.py                      # CLI entry point — runs predictions
│   ├── verification_engine.py       # Core verification logic (heuristic + VLM)
│   ├── dashboard_server.py          # FastAPI backend server
│   ├── index.html                   # Interactive dashboard (Tailwind CSS)
│   ├── generate_static_data.py      # Compiles claims_data.json for static deploy
│   ├── claims_data.json             # Pre-compiled data for Vercel (no backend)
│   └── evaluation/
│       ├── main.py                  # Evaluation script against sample_claims.csv
│       └── evaluation_report.md     # Metrics, cost analysis, and strategy comparison
└── dataset/
    ├── claims.csv                   # 44 test claims (no labels) — system input
    ├── sample_claims.csv            # 20 labeled examples for evaluation
    ├── user_history.csv             # Per-user claim counts and risk flags
    ├── evidence_requirements.csv    # Minimum image evidence rules by object type
    ├── output.csv                   # System predictions for claims.csv
    └── images/
        ├── sample/                  # Images for sample_claims.csv
        └── test/                    # Images for claims.csv
```

---

## How it works

### Four verification strategies

| Strategy | Model | Key needed | Cost |
|---|---|---|---|
| A — Heuristic | OpenCV + rule-based NLP | None | Free |
| B — GPT-4o VLM | OpenAI GPT-4o vision | `OPENAI_API_KEY` | ~$0.01/claim |
| C — Gemini 2.0 Flash | Google Gemini 2.0 Flash | `GOOGLE_API_KEY` | ~$0.001/claim |
| D — Qwen2.5-VL-72B | HuggingFace Serverless | `HUGGING_API_KEY` | Free tier |

All four are switchable from the dashboard dropdown without restarting the server.

### Risk scoring

Each claim gets a risk score (5–98) based on:

| Factor | Points |
|---|---|
| Base | 15 |
| High severity | +35 |
| Medium severity | +20 |
| Contradicted status | +25 |
| `possible_manipulation` flag | +30 |
| `text_instruction_present` flag | +25 |
| `user_history_risk` flag | +15 |
| `manual_review_required` flag | +10 |

---

## Output schema

Each row in `dataset/output.csv` contains:

| Column | Description |
|---|---|
| `user_id` | Submitting user |
| `image_paths` | Semicolon-separated image paths |
| `user_claim` | Original conversation transcript |
| `claim_object` | `car`, `laptop`, or `package` |
| `evidence_standard_met` | `true` / `false` |
| `evidence_standard_met_reason` | Short reason |
| `risk_flags` | Semicolon-separated flags or `none` |
| `issue_type` | `dent`, `crack`, `scratch`, `glass_shatter`, etc. |
| `object_part` | `front_bumper`, `screen`, `seal`, etc. |
| `claim_status` | `supported`, `contradicted`, `not_enough_information` |
| `claim_status_justification` | Image-grounded explanation |
| `supporting_image_ids` | `img_1;img_2` or `none` |
| `valid_image` | `true` / `false` |
| `severity` | `none`, `low`, `medium`, `high`, `unknown` |

---

## Setup and running

### Prerequisites

- Python 3.10+
- Tesseract OCR installed on your system ([install guide](https://github.com/UB-Mannheim/tesseract/wiki))
- An OpenAI API key (only needed for Strategy B / VLM)

### 1. Clone and install

```bash
git clone https://github.com/bikram73/MultiModal_Damage_Claim_Verification_System.git
cd MultiModal_Damage_Claim_Verification_System
pip install -r requirements.txt
```

### 2. Set your API keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...        # Strategy B — GPT-4o vision
GOOGLE_API_KEY=AIza...       # Strategy C — Gemini 2.0 Flash
HUGGING_API_KEY=hf_...       # Strategy D — Qwen2.5-VL-72B
```

All three are optional — Strategy A (heuristic) works with no keys at all. Each VLM strategy falls back to heuristic if its key is missing.

### 3. Run predictions

```bash
# Strategy A — Heuristic (no key needed)
python code/main.py heuristic

# Strategy B — GPT-4o vision (requires OPENAI_API_KEY)
python code/main.py vlm

# Strategy C — Gemini 2.0 Flash (requires GOOGLE_API_KEY)
python code/main.py gemini

# Strategy D — Qwen2.5-VL-72B via HuggingFace (requires HUGGING_API_KEY)
python code/main.py huggingface
```

Output is written to `dataset/output.csv`.

### 4. Run evaluation

```bash
python code/evaluation/main.py
```

Compares predictions against `dataset/sample_claims.csv` and writes results to `code/evaluation/evaluation_report.md`.

### 5. Launch the dashboard locally

```bash
python code/dashboard_server.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) — full interactive dashboard with claim browsing, image viewer, AI findings panel, risk scores, and CSV export.

---

## Dashboard features

- Browse all 64 claims (44 test + 20 sample) with search and filter
- Click any claim to see submitted images, verification report, and decision justification
- Switch between Strategy A and B and re-run the engine live
- Export filtered predictions as CSV
- Fully responsive — works on mobile and desktop
- Static fallback mode for Vercel deployment (no backend required)

---

## Deployment

The dashboard runs in two modes:

**Local (FastAPI)** — full backend with live prediction runs via `dashboard_server.py`

**Vercel (static)** — `vercel.json` routes everything to `index.html`, which auto-detects the cloud environment and loads `code/claims_data.json` instead of hitting the API

To deploy to Vercel, import the GitHub repo at [vercel.com](https://vercel.com) — no extra configuration needed.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Image analysis | OpenCV, Pillow, pytesseract |
| VLM — Strategy B | OpenAI GPT-4o |
| VLM — Strategy C | Google Gemini 2.0 Flash |
| VLM — Strategy D | Qwen2.5-VL-72B (HuggingFace) |
| Frontend | HTML, Tailwind CSS, vanilla JS |
| Deployment | Vercel (static), uvicorn (local) |
| Env management | python-dotenv |
