# NASA C-MAPSS Turbofan Predictive Maintenance & Explainable AI

An end-to-end Machine Learning, Explainable AI (XAI), and LLM diagnostic pipeline for turbofan engine Remaining Useful Life (RUL) estimation. Built for the IBM Datathon and optimized for LinuxONE (`s390x`) architecture.

---

## Architecture Overview

```
Raw Telemetry (C-MAPSS FD001)
       │
       ▼
[01_load_and_features.py]  ──► Rolling aggregations, baselines & unit-based split
       │
       ▼
[02_train.py]              ──► XGBoost Regressor (Test RMSE: 11.82, NASA Score: 179.1)
       │
       ▼
[03_explain.py]            ──► TreeSHAP attributions & fleet risk ranking
       │
       ▼
[generate_report.py]       ──► Groq LLM (qwen/qwen3.8-27b) diagnostic generation
       │
       ▼
[validate.py]              ──► 100% Anti-hallucination & numerical verification
```

---

## Key Performance Metrics

- **Test RMSE**: 11.82 cycles (vs. naive baseline: 43.00 cycles)
- **Test MAE**: 8.65 cycles (vs. naive baseline: 35.90 cycles)
- **NASA Asymmetric Score**: 179.1 (vs. naive baseline: 32,772.2)
- **Fleet Ranking**: Categorized into Critical (< 30 cycles), Warning (30–60 cycles), and Healthy (≥ 60 cycles)

---

## Repository Structure

```
├── reports/                     # Generated LLM maintenance diagnostic reports
│   ├── engine_081.md            # Critical status engine report
│   ├── engine_011.md            # Warning / median engine report
│   └── engine_047.md            # Healthy status engine report
├── scripts/
│   ├── 01_load_and_features.py  # Feature engineering & temporal rolling stats
│   ├── 02_train.py              # XGBoost training & JSON model persistence
│   ├── 03_explain.py            # TreeSHAP feature attributions
│   ├── common.py                # C-MAPSS parsing & endianness-safe I/O
│   ├── env_check.py             # Pre-flight environment & smoke tests
│   ├── generate_report.py       # Groq LLM report generator with reflection
│   └── setup_vm.sh              # LinuxONE s390x VM bootstrap script
├── outputs/
│   ├── explanations/            # Per-engine JSON dossiers (100 test engines)
│   ├── fleet.csv                # Urgency ranking table
│   ├── global_importance.json   # Fleet-wide sensor degradation hierarchy
│   ├── demo_engines.json        # Benchmark units (81, 11, 47)
│   └── metrics.json             # Model benchmark numbers
├── validate.py                  # Anti-hallucination verification test suite
├── requirements.txt             # Core Python dependencies
└── ENV_NOTES.md                 # LinuxONE environment diagnostic report
```

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/Annie578/IBM-Datathon-.git
cd IBM-Datathon-
pip install -r requirements.txt
```

### 2. Configure Groq API (Optional for LLM Generation)
Copy `.env.example` to `.env` and insert your Groq API key:
```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...
```

### 3. Generate Diagnostic Reports
```bash
# Generate report for critical engine #81
python scripts/generate_report.py --engine 81

# Generate all demo reports (81, 11, 47)
python scripts/generate_report.py --all
```

### 4. Run Anti-Hallucination Governance Validation
```bash
python validate.py
```
This strictly verifies that:
- Every number in the report matches the telemetry JSON.
- Only documented sensors from `top_drivers` are cited.
- True RUL is never leaked or mentioned.