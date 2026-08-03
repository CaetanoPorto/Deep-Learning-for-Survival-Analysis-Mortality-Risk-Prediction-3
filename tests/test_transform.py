"""Testes do pré-processamento (Etapa 5): fit só no treino, sem indicador de missing,
sem coluna de alvo, tolerante a categoria nova no teste."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.checks.leakage import assert_no_target_leakage
from src.config import NUMERIC_FEATURES, STAGE_ORDER
from src.preprocessing.transform import INPUT_COLUMNS, fit_transform_splits


def _frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "node_count": rng.integers(0, 20, n).astype(float),
        "nodes_examined_n": rng.integers(0, 30, n).astype(float),
        "tumor_size_mm": rng.integers(0, 60, n).astype(float),
        "age_mid": rng.choice([42, 52, 62, 72], n).astype(float),
        "stage": pd.Categorical(rng.choice(STAGE_ORDER, n), categories=STAGE_ORDER, ordered=True),
        "tumor_grade": rng.choice(["1", "2", "3", "4"], n),
        "node_status": rng.choice(["negativo", "positivo", "nao_avaliado", "desconhecido"], n),
        "race": rng.choice(["White", "Black", "Asian or Pacific Islander"], n),
        "er_status": rng.choice(["Positive", "Negative"], n),
        "pr_status": rng.choice(["Positive", "Negative"], n),
        "her2_status": rng.choice(["Positive", "Negative"], n),
        "surgery_group": rng.choice(["nenhuma", "conservadora", "mastectomia", "outra_desconhecida"], n),
        "radiation_group": rng.choice(["nenhuma_desconhecida", "feixe"], n),
        "chemotherapy": rng.choice(["Yes", "No/Unknown"], n),
    })


def test_input_columns_excluem_alvo_e_era():
    for proibido in ("duration", "event", "era_diagnostico", "breast_subtype"):
        assert proibido not in INPUT_COLUMNS


def test_fit_so_no_treino():
    df = _frame()
    pre = fit_transform_splits(df.iloc[:30], df.iloc[30:], df.iloc[30:])
    # colunas numéricas (as 4 primeiras) escaladas pelas estatísticas DO TREINO -> média ~0.
    assert np.abs(pre.X_train[:, :len(NUMERIC_FEATURES)].mean(axis=0)).max() < 1e-5


def test_sem_indicador_de_missing_e_sem_alvo():
    df = _frame()
    pre = fit_transform_splits(df.iloc[:30], df.iloc[30:], df.iloc[30:])
    assert not any("missingindicator" in c.lower() for c in pre.feature_names)
    assert_no_target_leakage(pre.feature_names)  # não levanta


def test_tolera_categoria_nova_no_teste():
    df = _frame()
    train = df.iloc[:30].copy()
    val = df.iloc[30:].copy()
    val.loc[val.index[0], "race"] = "American Indian/Alaska Native"  # ausente no treino
    pre = fit_transform_splits(train, val, val)  # handle_unknown="ignore" -> não quebra
    assert pre.X_val.shape[0] == len(val)
