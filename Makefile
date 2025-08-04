# Top-level Makefile for reproducible workflow

ENV_NAME?=waterbridge_mm

.PHONY: env test analysis sensitivity report pdf all clean

env:
	conda env create -f environment.yml -n $(ENV_NAME) || true

# ------------------------------------------------------------------
# Quality & tests
# ------------------------------------------------------------------

test:
	pytest -q

sensitivity:
	python code/generate_sensitivity.py

analysis:
	python code/analysis.py

# ------------------------------------------------------------------
# Build LaTeX report (requires TeX Live / MiKTeX in PATH)
# ------------------------------------------------------------------

report: sensitivity
	pdflatex -interaction=nonstopmode -output-directory=docs docs/final_comprehensive_report.tex

pdf: report  ## alias

all: test analysis report

clean:
	rm -f docs/*.aux docs/*.log docs/*.out docs/*.toc docs/*.gz docs/*_analysis.png
	rm -f docs/*.pdf docs/sensitivity_appendix.tex
