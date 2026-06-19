# 🛡️ ClaimVision AI - Multi-Modal Claim Verification Suite

ClaimVision AI is an enterprise-grade AI-powered claim verification platform designed to automate and streamline the first-level review of physical damage claims. By analyzing conversation transcripts, image files, policyholder history, and specific evidence rules, ClaimVision AI classifies claims into **Supported**, **Contradicted**, or **Not Enough Information** decisions in seconds.

---

## 🚀 Key Features

* **🧠 Smart Claim Extraction:** Parses chat transcripts using conversational keywords and semantic cues to extract the target object, part, and claimed damage.
* **📸 Multi-Modal Image Verification:** Uses OpenCV and Pillow to run physical image checks:
  * *Blurry Images:* Variance of Laplacian blur check.
  * *Low Light & Glare:* Average pixel brightness detection.
  * *Prompt Injection / Fraud:* Optical Character Recognition (OCR via Pytesseract) checking for text note instructions attempting to override standard review.
* **📏 Evidence Requirement Validation:** Checks claim variables against standard checklists (e.g. car glass visibility, package interior contents unboxing verification) from `evidence_requirements.csv`.
* **📊 User History Risk Analyzer:** Cross-references the claimant's ID against historical claims in `user_history.csv` to compute an aggregated claimant risk profile.
* **⚡ Dual-Mode Enterprise Dashboard:** A Stripe and Notion-inspired interface that runs in two configurations:
  * **Dynamic (FastAPI) Mode:** Connects to a live FastAPI Python backend on port 8000 for running prediction strategies on the fly.
  * **Static (Vercel) Mode:** Automatically falls back to loading pre-compiled `claims_data.json` statically, rendering all features without any backend requirements.

---

## 📂 Repository File Structure

```text
Modern_Enterprise_AI_Dashboard/
├── vercel.json                         # Vercel static deployment routing config
├── README.md                           # Main documentation file (You are here)
├── AGENTS.md                           # Onboarding rules and chat log registry
├── code/                               # Application source code
│   ├── main.py                         # Command-line prediction engine runner
│   ├── verification_engine.py          # Core claims classifier & image quality logic
│   ├── dashboard_server.py             # FastAPI backend REST API server
│   ├── index.html                      # Interactive Tailwind CSS dashboard template
│   ├── generate_static_data.py         # Script compiling claims data to static JSON
│   ├── claims_data.json                # Pre-compiled claims data for Vercel runs
│   └── evaluation/
│       ├── main.py                     # Evaluation script
│       └── evaluation_report.md        # Compiled metrics comparison report
└── dataset/                            # Input CSV files and image databases
    ├── sample_claims.csv               # Ground-truth labeled samples
    ├── claims.csv                      # Test input rows for prediction
    ├── user_history.csv                # Claimant histories and flag indexes
    ├── evidence_requirements.csv       # Minimum evidence checklists
    └── images/
        ├── sample/                     # Images referenced by sample_claims.csv
        └── test/                       # Images referenced by claims.csv
```

---

## 🛠️ Installation and Setup

### Prerequisites
Make sure you have **Python 3.10+** and `pip` installed on your machine.

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd Modern_Enterprise_AI_Dashboard
```

### 2. Install dependencies
Install the required packages using pip:
```bash
pip install fastapi uvicorn pillow opencv-python pytesseract numpy pydantic
```
*(Optional: Install `pytesseract` system binaries for image-to-text OCR check features).*

---

## 📖 How to Use

### 1. Run predictions on test dataset
Process `dataset/claims.csv` and write outputs in the exact required schema to `dataset/output.csv`:
```bash
python code/main.py [strategy]
```
*Available strategies:* `heuristic` (default), `vlm`

### 2. Run system evaluations
Compare strategies against labeled ground truth in `dataset/sample_claims.csv` and output accuracy metrics:
```bash
python code/evaluation/main.py
```
This generates the full [evaluation_report.md](code/evaluation/evaluation_report.md).

### 3. Launch the dashboard server locally
Start the local FastAPI development server:
```bash
python code/dashboard_server.py
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser. You can:
* Search/filter claims by ID, object, or user.
* Click rows to dynamically update the images and AI findings details.
* Switch prediction strategies and execute model runs.

---

## 🛠️ Technical Details & Implementation Architecture

### 1. Multi-Modal Image Validation Algorithms 📸
The system uses **OpenCV** and **Pillow** to perform automated checks on the submitted image evidence:
* **Blurriness Check (Laplacian Variance):** Applies a Laplace kernel filter to calculate the variance of focal sharpness. If the variance is `< 80.0`, the image is flagged as `blurry_image`.
* **Low-Light / Glare Detection:** Converts images to grayscale and computes the mean luminance. A mean brightness `< 35.0` (out of 255) triggers `low_light_or_glare`.
* **Prompt Injection Detection (OCR):** Uses `pytesseract` to scan the image for printed notes or text blocks containing instructions such as *"approve the claim"*, *"ignore previous checks"*, or *"override"*. If found, it flags `text_instruction_present` and routes the claim to manual review/contradiction.

### 2. Multi-Variant Claimant Risk Score 📊
For each claim, a dynamic **Risk Score (5 to 98)** is computed using the following weights:
* **Base Risk:** `15` points.
* **Claim Severity:** `+35` for High severity claims, `+20` for Medium severity, `+10` for Low severity.
* **Verification Status:** `+25` if the visual evidence contradicts the claim conversation; `+10` if evidence is ambiguous or missing.
* **Risk Flags triggered:**
  * `possible_manipulation`: `+30` points.
  * `text_instruction_present`: `+25` points.
  * `user_history_risk`: `+15` points.
  * `manual_review_required`: `+10` points.

### 3. Rule-Based Natural Language Processing 🧠
The parsing engine extracts the target component and damage category from conversation threads using token matching:
* **Part Extractor:** Maps keywords like *windshield*, *fender*, *hinge*, *trackpad*, *keyboard*, *seal*, *label*, *mirror* directly to standard categories.
* **Issue Extractor:** Identifies damage families like *dent*, *scratch*, *crack*, *glass_shatter*, *missing_part*, *torn_packaging*, *water_damage*, and *stain*.

---

## ☁️ GitHub & Vercel Deployment

ClaimVision AI is pre-configured to be deployed as a static web app on Vercel without requiring a live Python runtime server.

### Deploying to GitHub
1. Initialize Git and add your files:
   ```bash
   git init
   git add .
   git commit -m "feat: implement ClaimVision AI system and dashboard"
   ```
2. Link your local repository to GitHub and push:
   ```bash
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

### Deploying to Vercel
1. Log in to [Vercel](https://vercel.com) and click **Add New Project**.
2. Import your GitHub repository.
3. Vercel will automatically read `vercel.json` and deploy your dashboard.
4. The dashboard will auto-detect the cloud environment, load the pre-compiled `claims_data.json` database, and render all metrics, images, and details correctly in the cloud!

