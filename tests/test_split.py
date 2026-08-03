"""Testes do split estratificado por event × era (Etapa 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ERA_LABELS
from src.data.era import ERA_COLUMN
from src.preprocessing.split import train_val_test_split


def _synthetic(n: int = 6000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    era_vals = rng.choice(ERA_LABELS, size=n, p=[0.15, 0.24, 0.26, 0.10, 0.25])
    # taxa de evento depende da era (como na base real) — o que torna a estratificação
    # por era necessária para equilibrar a censura entre folds.
    era_rate = dict(zip(ERA_LABELS, [0.59, 0.47, 0.31, 0.20, 0.10]))
    event = (rng.random(n) < np.array([era_rate[e] for e in era_vals])).astype(int)
    return pd.DataFrame(
        {
            ERA_COLUMN: pd.Categorical(era_vals, categories=ERA_LABELS, ordered=True),
            "event": event,
            "duration": rng.integers(0, 200, n).astype(float),
            "feat": rng.normal(size=n),
        }
    )


def test_fracoes_e_sem_perda():
    df = _synthetic()
    s = train_val_test_split(df)
    assert len(s.train) + len(s.val) + len(s.test) == len(df)
    assert abs(len(s.train) / len(df) - 0.70) < 0.01
    assert abs(len(s.val) / len(df) - 0.15) < 0.01


def test_estratificacao_equilibra_era_e_censura():
    df = _synthetic()
    s = train_val_test_split(df)
    assert s.max_era_gap < 0.03  # proporção de era quase igual nos 3 folds
    assert s.max_censoring_gap < 0.03  # censura equilibrada apesar de depender da era


def test_reprodutivel_com_mesma_semente():
    df = _synthetic()
    a = train_val_test_split(df, seed=42)
    b = train_val_test_split(df, seed=42)
    pd.testing.assert_frame_equal(a.train, b.train)


def test_split_nao_altera_valores():
    # o split só particiona linhas — nenhum fit/transform. O conjunto de valores de uma
    # feature é preservado ao reunir os folds.
    df = _synthetic()
    s = train_val_test_split(df)
    juntos = pd.concat([s.train["feat"], s.val["feat"], s.test["feat"]])
    assert sorted(juntos.tolist()) == sorted(df["feat"].tolist())
