"""Advanced econometric routines that back specific forecasts.

Functions here are data-driven, rely only on lightweight dependencies
(statsmodels & scikit-learn) and read snapshot CSVs shipped in the repo.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from sklearn.linear_model import LogisticRegression

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

# -----------------------------------------------------------------------------
# F17 & F18 – Reserve-Currency Shares via VAR
# -----------------------------------------------------------------------------

def var_reserve_shares(year_forecast: int = 2030, usd_threshold: float = 55.0, rmb_threshold: float = 7.0) -> dict:
    """Fit a VAR(2) on USD and RMB reserve shares.

    Returns a dict with keys 'usd_prob_gt_50' and 'rmb_prob_lt_10'.
    """
    df = pd.read_csv(DATA_DIR / "forex_reserves.csv")
    df = df.set_index("Year")

    model = VAR(df[["USD_Share", "RMB_Share"]])
    results = model.fit(maxlags=2, ic='aic')

    # Forecast point estimate + Monte Carlo draws for uncertainty   
    steps = year_forecast - df.index.max()
    point_forecast = results.forecast(df.values[-results.k_ar:], steps)[-1]

    usd_point, rmb_point = point_forecast

    # Very light bootstrap for variance (100 draws)
    # statsmodels <=0.13 simulate_var API returns (steps, k) without replic support;
    # as a lightweight approximation, sample residual covariance to generate 200 draws.
    cov = results.sigma_u
    simulated = np.random.multivariate_normal(mean=point_forecast, cov=cov, size=200)
    usd_draws = simulated[:, 0]
    rmb_draws = simulated[:, 1]

    prob_usd = float((usd_draws > usd_threshold).mean())
    prob_rmb = float((rmb_draws < rmb_threshold).mean())

    return {
        "usd_prob_gt_50": prob_usd,
        "usd_point": usd_point,
        "rmb_prob_lt_10": prob_rmb,
        "rmb_point": rmb_point,
    }

# -----------------------------------------------------------------------------
# F20 – Carbon-Tariff Proliferation via Logistic Adoption Curve
# -----------------------------------------------------------------------------

def logistic_carbon_tariffs(year_target: int = 2029, threshold: int = 5) -> float:
    """Estimate probability that >=3 G-20 economies adopt carbon tariffs."""
    df = pd.read_csv(DATA_DIR / "carbon_tariffs.csv")  # Year, AdoptedCount
    X = (df["Year"] - df["Year"].min()).values.reshape(-1, 1)
    y = (df["AdoptedCount"] >= threshold).astype(int).values

    clf = LogisticRegression()
    clf.fit(X, y)

    target_x = np.array([[year_target - df["Year"].min()]])
    prob = float(clf.predict_proba(target_x)[0, 1])
    return prob

# -----------------------------------------------------------------------------
# F14 – Technology Standards Bifurcation via Bass Diffusion (Logistic surrogate)
# -----------------------------------------------------------------------------

def logistic_tech_bifurcation(year_target: int = 2028, threshold: int = 5) -> float:
    """Probability >=3 verticals show bifurcation.

    Input file: tech_standards.csv with columns Year, BifurcatedVerticals.
    """
    df = pd.read_csv(DATA_DIR / "tech_standards.csv")
    X = (df["Year"] - df["Year"].min()).values.reshape(-1, 1)
    y = (df["BifurcatedVerticals"] >= threshold).astype(int)

    clf = LogisticRegression()
    clf.fit(X, y)

    target_x = np.array([[year_target - df["Year"].min()]])
    prob = float(clf.predict_proba(target_x)[0, 1])
    return prob


if __name__ == "__main__":
    print(var_reserve_shares())
    print(logistic_carbon_tariffs())
    print(logistic_tech_bifurcation())
