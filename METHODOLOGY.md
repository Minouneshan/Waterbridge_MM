# Methodology — Modern Mercantilism Forecasting

> *A transparent, reproducible framework that blends Bayesian belief networks, advanced econometrics, and game-theoretic reasoning to quantify the economic consequences of state-led competition.*

---

## 1. Executive Summary
Our methodology produces calibrated probability estimates for **25 binary forecasts** spanning 1–10 years. The framework integrates:

1. **Bayesian Belief Networks** – to propagate evidence across causally linked forecasts.
2. **Advanced Econometrics** – to model time-series dynamics and structural breaks.
3. **Strategic Game-Theory** – to capture incentive-driven policy interactions.
4. **Rigorous Source Weighting** – to combine heterogeneous evidence streams with traceable credibility scores.

---

## 2. Core Analytical Framework
### 2.1 Bayesian Belief Network
* **Implementation:** `code/bayesian_model.py`
* **Nodes:** 25 forecasts organised into four causal clusters (Trade-Security, Tech Competition, Financial Fragmentation, Resource-Climate).
* **Update Rule:**
\[\text{logit}(P_{\text{updated}})=\text{logit}(P_{\text{prior}})+\sum_i w_i E_i\]
with **w\_i ∈ [0.40,0.90]** reflecting source credibility and **E\_i** the signed evidence strength.

### 2.2 Econometric Toolkit
* Vector Autoregression (VAR) for coupled trends.
* Monte-Carlo simulation (10 000 draws) for uncertainty bounds.
* Structural-break testing (Chow & Bai-Perron) identifies regime shifts (e.g., 2018 tariff wave).
* Granger causality for directionality validation.

### 2.3 Game-Theoretic Modules
* **Industrial Subsidy Game:** N-player Nash equilibrium modelling subsidy escalation.
* **Technology Standards Race:** Network-effects model for competing ecosystems (5G/6G, AI, semiconductors).
* **Currency Competition:** Payoff matrix for reserve-currency fragmentation versus inertia.

---

## 3. Data Architecture & Evidence Weighting
| Tier | Typical Sources | Credibility Weight |
|------|-----------------|--------------------|
| **1** | IMF, WTO, ECB, U.S. Census FT-900 | **0.90** |
| **2** | OECD, World Bank, BIS, SIA | **0.80** |
| **3** | Brookings, PIIE, McKinsey, OIES | **0.70** |

*Cross-Source Verification, Temporal Decay (−5 % per year), and Conflict-of-Interest screening guard against bias.*

---

## 4. Quality Assurance
* **Calibration:** Brier score tracking with target zone ±0.05.
* **Back-testing:** Validation against 2008–2022 hind-cast.
* **Sensitivity Analysis:** Parameter, structural, and data perturbations.
* **Peer Review:** External domain experts interrogate assumptions bi-annually.

---

## 5. Transparency & Reproducibility
* **Code:** Fully documented Python in `code/`.
* **Data Provenance:** `data/source_links.txt` enumerates all raw URLs.
* **Assumptions:** Each forecast lists explicit resolution criteria and causal pathways in `docs/final_comprehensive_report.pdf`.
* **Continuous Updating:** New evidence triggers scheduled probability refresh via CI pipeline.

---

## 6. Limitations & Future Work
1. **Historical Sample Size:** Limited precedents for technology bifurcation.
2. **Political Variables:** Measurement error remains high; exploring NLP extraction of legislative texts.
3. **Horizon Risk:** Forecast confidence decays beyond five years; scenario analysis supplements point estimates.
4. **Computation:** Scaling Bayesian network beyond 25 nodes demands optimisation (planned GPU port).

Planned enhancements include **real-time news ingestion** via transformer-based classifiers and **international model collaboration** for cross-validation.

---

*Last updated: 2025-08-04*
