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

# 3. Run the test-suite (optional but recommended)
pytest -q
```

### 2.3 Building the report
```bash
# Compile the LaTeX document (output goes to docs/)
make pdf         # convenient wrapper – or run the pdflatex sequence below

# Manual alternative
pdflatex  -interaction=nonstopmode -output-directory=docs  docs/final_comprehensive_report.tex
```

### 2.4 Reproducing the analysis
```bash
python code/index_calc.py         # BRICS & monetary indicators
python code/simple_analysis.py    # Core regression & trend analysis
python code/analysis.py           # Full Bayesian & game-theoretic model
```
All scripts write intermediate CSVs/PNGs to the `data/` and `docs/` directories and can be chained in a single run-all workflow (see `code/analysis.py`).

---

## 3. Key Findings (selected)
* **WTO Appellate Body stall (92 %):** high persistence of unilateral trade remedies until at least 2027.
* **Supply-chain pivot (78 %):** China’s share of U.S. goods imports falls below **12 %** by 2027; Vietnam trade volume with the U.S. doubles (74 %).
* **Tech standards bifurcation (86 %):** Distinct U.S.- versus China-led standards dominate **≥ 3** critical technology verticals by 2030.
* **Industrial subsidy race (80 %):** ≥ 5 G-20 economies announce subsidy programmes exceeding **$50 bn** before 2026.

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
