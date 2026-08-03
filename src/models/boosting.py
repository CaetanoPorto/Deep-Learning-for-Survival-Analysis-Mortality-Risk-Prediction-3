"""Gradient Boosting de sobrevivência (scikit-survival) — baseline não linear que
costuma ser o mais forte dos três clássicos (Plano de Modelagem).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from src.config import GBSConfig, RANDOM_SEED


def fit_gbs(X_train: np.ndarray, y_train: np.ndarray, config: GBSConfig = GBSConfig()) -> GradientBoostingSurvivalAnalysis:
    model = GradientBoostingSurvivalAnalysis(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def gbs_risk(model: GradientBoostingSurvivalAnalysis, X: np.ndarray) -> np.ndarray:
    """Escore de risco (log-hazard relativo): maior = maior risco."""
    return np.asarray(model.predict(X))


def gbs_survival(model: GradientBoostingSurvivalAnalysis, X: np.ndarray, batch: int = 5_000) -> pd.DataFrame:
    """Curva S(t) por indivíduo (índice = tempos do modelo, colunas = indivíduos)."""
    X = np.asarray(X)
    times = getattr(model, "unique_times_", None)
    if times is None:
        times = model.event_times_
    parts = [model.predict_survival_function(X[i:i + batch], return_array=True) for i in range(0, len(X), batch)]
    return pd.DataFrame(np.vstack(parts).T, index=times)
