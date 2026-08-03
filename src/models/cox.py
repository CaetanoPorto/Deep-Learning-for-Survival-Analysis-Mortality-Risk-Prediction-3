"""Baselines Cox (lifelines): Cox proporcional com ridge, teste de proporcionalidade de
Schoenfeld e Cox + splines nas contínuas.

O Cox estabelece o piso obrigatório: se a rede não bater o Cox, não há TCC. O teste de
proporcionalidade precisa ser feito (não assumido); com N enorme ele quase sempre rejeita,
então o que importa é o TAMANHO do desvio, não só o p-valor (Plano de Modelagem).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.preprocessing import SplineTransformer

from src.config import CoxConfig, SplineConfig


def _sanitize(names: list[str]) -> list[str]:
    """Nomes de coluna seguros para o lifelines (sem espaços/caracteres especiais)."""
    return [re.sub(r"[^0-9a-zA-Z_]+", "_", n) for n in names]


def build_cox_frame(X: np.ndarray, feature_names: list[str], duration, event) -> pd.DataFrame:
    frame = pd.DataFrame(np.asarray(X), columns=_sanitize(feature_names))
    frame["duration"] = np.asarray(duration, dtype=float)
    frame["event"] = np.asarray(event, dtype=int)
    return frame


def fit_cox(train_frame: pd.DataFrame, config: CoxConfig = CoxConfig()) -> CoxPHFitter:
    """Ajusta o CoxPHFitter (ridge, empates de Efron por padrão do lifelines — necessário
    com muitos empates em t=0, ADR-004).

    Duração 0 é deslocada para 0,5 mês SÓ no ajuste (o lifelines rejeita duração não
    positiva). O C-index é avaliado com as durações ORIGINAIS e o risco só depende das
    covariáveis, então isso não afeta a métrica.
    """
    df = train_frame.copy()
    df["duration"] = df["duration"].clip(lower=0.5)
    model = CoxPHFitter(penalizer=config.penalizer)
    model.fit(df, duration_col="duration", event_col="event")
    return model


def cox_risk(model: CoxPHFitter, frame: pd.DataFrame) -> np.ndarray:
    """Escore de risco (partial hazard): maior = maior risco."""
    covariates = frame.drop(columns=["duration", "event"])
    return model.predict_partial_hazard(covariates).to_numpy()


def cox_survival(model: CoxPHFitter, frame: pd.DataFrame) -> pd.DataFrame:
    """Curva S(t) por indivíduo (índice = tempos, colunas = indivíduos)."""
    covariates = frame.drop(columns=["duration", "event"])
    return model.predict_survival_function(covariates)


@dataclass
class PHSummary:
    n_covariates: int
    n_violating_05: int
    n_violating_01: int
    top: pd.DataFrame  # maiores estatísticas de teste


def proportional_hazards_summary(model: CoxPHFitter, train_frame: pd.DataFrame) -> PHSummary:
    """Teste de Schoenfeld. Roda no MESMO frame de treino usado no ajuste (o lifelines
    exige N igual ao do fit). Com N grande quase tudo rejeita — reportamos quantas
    covariáveis violam e as maiores estatísticas, para olhar o TAMANHO do desvio.
    """
    df = train_frame.copy()
    df["duration"] = df["duration"].clip(lower=0.5)
    test = proportional_hazard_test(model, df, time_transform="rank")
    summ = test.summary
    return PHSummary(
        n_covariates=len(summ),
        n_violating_05=int((summ["p"] < 0.05).sum()),
        n_violating_01=int((summ["p"] < 0.01).sum()),
        top=summ.sort_values("test_statistic", ascending=False).head(5),
    )


def build_spline_features(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
    feature_names: list[str], config: SplineConfig = SplineConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Expande as colunas contínuas (num__*) em bases de spline cúbica (fit SÓ no treino)
    e mantém as demais (ordinais + one-hot) como estão.
    """
    num_idx = [i for i, n in enumerate(feature_names) if n.startswith("num__")]
    other_idx = [i for i in range(len(feature_names)) if i not in num_idx]

    st = SplineTransformer(n_knots=config.n_knots, degree=config.degree, include_bias=False)
    tr = st.fit_transform(X_train[:, num_idx])
    va = st.transform(X_val[:, num_idx])
    te = st.transform(X_test[:, num_idx])

    Xtr = np.hstack([tr, X_train[:, other_idx]]).astype(np.float32)
    Xva = np.hstack([va, X_val[:, other_idx]]).astype(np.float32)
    Xte = np.hstack([te, X_test[:, other_idx]]).astype(np.float32)

    names = [f"sp_{i}" for i in range(tr.shape[1])] + [feature_names[j] for j in other_idx]
    return Xtr, Xva, Xte, names
