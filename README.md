# Modern Mercantilism — Global Forecasting Analysis

> *A quantitative & qualitative research project forecasting the economic, technological, and geopolitical repercussions of the emerging era of **Modern Mercantilism***.

---

## 1. Project Overview
This repository contains all code, data, documentation, and LaTeX sources required to reproduce our **25 binary forecasts** covering a 1 – 10-year horizon. Forecast topics span trade institutions, supply-chain realignment, technology bifurcation, monetary fragmentation, and industrial policy competition.

The flagship deliverable is the peer-review-ready PDF report `docs/final_comprehensive_report.pdf`, supported by fully reproducible Python analysis and transparent data provenance.

## 2. Quick-start
### 2.1 Prerequisites
* **Conda ≥ 22** (or **Mamba**) for environment management  
* **TeX Live** or **MiKTeX** for LaTeX compilation  
* **Python ≥ 3.11** (automatically installed via the conda environment)

### 2.2 Setup
```bash
# 1. Create the environment
a) conda env create -f environment.yml      # or: mamba env create -f environment.yml

# 2. Activate it
conda activate waterbridge_mm

# 3. Install additional packages (if needed)
pip install pgmpy arch linearmodels seaborn

# 4. Run comprehensive analysis
python code/analysis.py

# 5. Run statistical validation (optional)
python code/statistical_tests.py
```

### 2.3 Building the report
```bash
# Compile the LaTeX documents (output goes to docs/)
make pdf         # convenient wrapper – or run the pdflatex sequence below

# Manual alternative
pdflatex  -interaction=nonstopmode -output-directory=docs  docs/final_comprehensive_report.tex
```

### 2.4 Reproducing the analysis
```bash
python code/analysis.py                # Full comprehensive analysis
python code/advanced_bayesian_model.py # Test Bayesian network implementation
python code/statistical_tests.py       # Econometric validation tests
python code/generate_sensitivity.py    # Generate sensitivity analysis
```
All scripts write intermediate CSVs/PNGs to the `data/` and `docs/` directories and can be chained in a single run-all workflow (see `code/analysis.py`).

---

## 3. Key Findings (selected)
* **Tariff escalation (70%):** Global average MFN tariffs rise by ≥2 percentage points above 2022 baseline by 2026.
* **Supply-chain pivot (75%):** China's share of U.S. goods imports falls below **12%** by 2027; Vietnam trade volume with the U.S. doubles (72%).
* **Tech standards bifurcation (85%):** Distinct U.S.- versus China-led standards dominate **≥5** critical technology verticals by 2027.
* **USD reserve resilience (66%):** U.S. dollar maintains **>55.5%** of global FX reserves through 2030 despite fragmentation.
* **Carbon tariff proliferation (64%):** **≥7** G-20 economies implement carbon tariffs by 2029.

(See the report for the full list of 25 forecasts, evidence, and resolution criteria.)

---

## 4. Repository Map
```
Waterbridge_MM/
├── code/                # Python analysis & modelling scripts
├── data/                # Input datasets & generated intermediates
├── docs/                # Final report, figures, calibration CSVs
├── tests/               # PyTest unit & regression tests
├── environment.yml      # Conda environment specification
└── README.md & METHODOLOGY.md
```

---

## 5. Contributing & Issues
Pull requests improving documentation, tests, or analytical methods are welcome. Please open an issue for substantive methodological changes before submitting a PR.

---

## 6. License
Distributed under the **MIT License**. See `LICENCE` for full text.

---
*Last updated: 2025-08-04*
