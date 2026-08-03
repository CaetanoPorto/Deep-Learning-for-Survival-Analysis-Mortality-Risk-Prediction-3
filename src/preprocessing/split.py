"""Etapa 4 do RUNBOOK — split treino/validação/teste.

Sempre ANTES de qualquer `fit` (regra anti-vazamento). Estratifica por `event` **E**
`era_diagnostico` — não só por `event` (essa é a correção vs. código antigo). Se a era
não entrar na estratificação, os folds ficam com composição de era desigual e a taxa de
censura diverge entre eles (a era determina o follow-up, logo a censura). Ver ADR-003 e
Protocolo de Validação.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CENSORING_RATE_TOLERANCE, RANDOM_SEED, TEST_FRAC, TRAIN_FRAC, VAL_FRAC
from src.data.era import ERA_COLUMN
from src.data.target import EVENT_COL


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    censoring_rates: dict[str, float]
    max_censoring_gap: float
    era_proportions: dict[str, dict[str, float]] = field(default_factory=dict)
    max_era_gap: float = 0.0


def _strata(df: pd.DataFrame) -> pd.Series:
    """Rótulo de estrato = event × era (10 estratos: 2 eventos × 5 eras)."""
    return df[EVENT_COL].astype(str) + "|" + df[ERA_COLUMN].astype(str)


def train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = RANDOM_SEED,
) -> SplitResult:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, "as frações devem somar 1.0"

    train_df, rest_df = train_test_split(
        df, train_size=train_frac, stratify=_strata(df), random_state=seed
    )
    val_rel = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        rest_df, train_size=val_rel, stratify=_strata(rest_df), random_state=seed
    )

    folds = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }

    censoring_rates = {name: float(1 - f[EVENT_COL].mean()) for name, f in folds.items()}
    max_censoring_gap = max(censoring_rates.values()) - min(censoring_rates.values())

    era_props = {
        name: {era: float(p) for era, p in f[ERA_COLUMN].value_counts(normalize=True).items()}
        for name, f in folds.items()
    }
    eras = set().union(*[set(p) for p in era_props.values()])
    max_era_gap = max(
        max(era_props[n].get(e, 0.0) for n in folds) - min(era_props[n].get(e, 0.0) for n in folds)
        for e in eras
    )

    if max_censoring_gap > CENSORING_RATE_TOLERANCE:
        print(f"[split] AVISO: censura varia {max_censoring_gap:.2%} entre folds: {censoring_rates}")

    return SplitResult(
        train=folds["train"],
        val=folds["val"],
        test=folds["test"],
        censoring_rates=censoring_rates,
        max_censoring_gap=max_censoring_gap,
        era_proportions=era_props,
        max_era_gap=max_era_gap,
    )
