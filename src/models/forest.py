"""Random Survival Forest (scikit-survival) — baseline não linear sem rede neural.

Costuma ser competitivo com deep learning em dados tabulares; é um dos testes reais de
se "DL traz ganho" (Plano de Modelagem).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest

from src.config import RANDOM_SEED, RSFConfig


def fit_rsf(X_train: np.ndarray, y_train: np.ndarray, config: RSFConfig = RSFConfig()) -> RandomSurvivalForest:
    model = RandomSurvivalForest(
        n_estimators=config.n_estimators,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def rsf_risk(model: RandomSurvivalForest, X: np.ndarray, batch: int = 2_000) -> np.ndarray:
    """Escore de risco (soma do hazard cumulativo): maior = maior risco.

    Prediz em lotes E single-thread: o `predict` do sksurv materializa a curva de hazard
    de todas as linhas ((n, n_tempos, 2) float64) e, com n_jobs=-1, cada worker aloca a
    sua — estoura a memória em n grande. Lote pequeno + n_jobs=1 limita o pico sem mudar
    o resultado. (O `fit` continua paralelo.)
    """
    model.n_jobs = 1
    X = np.asarray(X)
    out = np.empty(len(X), dtype=float)
    for i in range(0, len(X), batch):
        out[i:i + batch] = model.predict(X[i:i + batch])
    return out


def rsf_survival(model: RandomSurvivalForest, X: np.ndarray, batch: int = 2_000) -> pd.DataFrame:
    """Curva S(t) por indivíduo (índice = tempos do modelo, colunas = indivíduos). Em
    lotes e single-thread pela mesma razão de memória de `rsf_risk`.
    """
    model.n_jobs = 1
    X = np.asarray(X)
    times = getattr(model, "unique_times_", None)
    if times is None:
        times = model.event_times_
    parts = [model.predict_survival_function(X[i:i + batch], return_array=True) for i in range(0, len(X), batch)]
    return pd.DataFrame(np.vstack(parts).T, index=times)
