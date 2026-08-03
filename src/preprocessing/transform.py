"""Etapa 5 do RUNBOOK — pré-processamento num único ColumnTransformer, ajustado
EXCLUSIVAMENTE no treino (regra anti-vazamento; o C-index 1.0 nasceu de um fit antes do
split — ver Vazamento de Dados).

Estrutura das 4 famílias de coluna:
  - numéricas -> mediana + StandardScaler (sem indicador de missing);
  - ordinais (stage, tumor_grade) -> OrdinalEncoder (ordem) + mediana + escala;
  - nominais -> categoria "desconhecido" + one-hot(handle_unknown="ignore"), incluindo
    her2_status (ADR-009): preserva o HER2 real na coorte principal; na coorte completa
    o "desconhecido" indica pré-2010, tratado por restrição de coorte, não por imputação.

O transformer recebe SÓ as colunas de feature: `duration`, `event` e `era_diagnostico`
nunca entram (impossível vazarem por construção).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    DESCONHECIDO_LEVEL,
    NOMINAL_DESCONHECIDO,
    NUMERIC_FEATURES,
    ORDINAL_ENCODE,
)

ORDINAL_COLS = list(ORDINAL_ENCODE)
# Colunas que ENTRAM no transformer — exclui duration/event/era/breast_subtype por não
# estarem nesta lista (o transformer nunca as vê).
INPUT_COLUMNS = NUMERIC_FEATURES + ORDINAL_COLS + NOMINAL_DESCONHECIDO


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    ordinal_pipe = Pipeline([
        ("ord", OrdinalEncoder(
            categories=[ORDINAL_ENCODE[c] for c in ORDINAL_COLS],
            handle_unknown="use_encoded_value", unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )),
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    desconhecido_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value=DESCONHECIDO_LEVEL)),
        ("oh", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False, dtype=np.float32)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("ord", ordinal_pipe, ORDINAL_COLS),
            ("nom", desconhecido_pipe, NOMINAL_DESCONHECIDO),
        ],
        remainder="drop",
    )


@dataclass
class PreprocessedSplits:
    transformer: ColumnTransformer
    feature_names: list[str]
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray


def fit_transform_splits(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> PreprocessedSplits:
    """Ajusta o ColumnTransformer SÓ no treino e aplica (transform) em val/teste."""
    ct = build_preprocessor()
    X_train = ct.fit_transform(train[INPUT_COLUMNS]).astype(np.float32)
    X_val = ct.transform(val[INPUT_COLUMNS]).astype(np.float32)
    X_test = ct.transform(test[INPUT_COLUMNS]).astype(np.float32)
    return PreprocessedSplits(
        transformer=ct,
        feature_names=list(ct.get_feature_names_out()),
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
    )
