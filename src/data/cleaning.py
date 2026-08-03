"""Etapa 2 do RUNBOOK — limpeza segundo as regras do SEER.

Ordem (Pipeline de Pré-processamento):
  1. harmonizar vocabulário + combine_first dos pares antigo/novo  (resolve Blank(s)
     estrutural — a era JÁ foi reconstruída, então isso é legítimo);
  2. mascarar tokens de nulo REAIS (Unknown, Borderline/Unknown, ...);
  3. decodificar sentinelas (linfonodos, tamanho — sentinels.py, ADR-005);
  4. derivar features agrupadas (idade, cirurgia, radiação, estadiamento — features.py).

Diferença vs. código antigo: `Blank(s)` NÃO está no conjunto de nulos globais (a era
seria destruída); a decodificação de sentinelas recupera 95/97 (nodo-positivo) e as
faixas 991-995 de tamanho, em vez de mascará-las como faltante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    BLANK_TOKEN,
    CATEGORICAL_FEATURES,
    COL_AGE,
    COL_BREAST_SUBTYPE,
    COL_CHEMOTHERAPY,
    COL_NODES_EXAMINED,
    COL_NODES_POSITIVE,
    COL_RACE,
    COL_RADIATION,
    COL_STAGE,
    COL_SURGERY,
    GRADE_NULL_TOKENS,
    KEPT_NOT_MODELED,
    NUMERIC_FEATURES,
    REAL_NULL_TOKENS,
    REDUNDANT_PAIRS,
)
from src.data.era import ERA_COLUMN
from src.data.features import clean_stage, decode_age_mid, group_radiation, group_surgery
from src.data.sentinels import (
    decode_node_count,
    decode_node_status,
    decode_nodes_examined,
    decode_tumor_size,
    tumor_size_diffuse_flag,
)
from src.data.target import DURATION_COL, EVENT_COL

pd.set_option("future.no_silent_downcasting", True)

LABEL_STRUCTURAL_COLS = [ERA_COLUMN, DURATION_COL, EVENT_COL]


def _mask(series: pd.Series, tokens: set[str]) -> pd.Series:
    """Substitui `tokens` por NaN sem downcast silencioso (isin + where)."""
    return series.where(~series.isin(tokens), other=np.nan)


def harmonize_and_combine_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica cada par antigo/novo em uma coluna (prioridade ao mais novo).

    `Blank(s)` vira NaN nos dois lados ANTES de combinar (a era já está salva), o que
    faz o combine escolher o valor real do outro lado; o vocabulário é harmonizado antes,
    senão "Positive" e "ER positive" viram níveis distintos.
    """
    df = df.copy()
    for pair in REDUNDANT_PAIRS:
        old = _mask(df[pair.old_col], {BLANK_TOKEN})
        new = _mask(df[pair.new_col], {BLANK_TOKEN})
        if pair.old_value_map:
            old = old.replace(pair.old_value_map)
        if pair.new_value_map:
            new = new.replace(pair.new_value_map)
        df[pair.output_col] = new.where(new.notna(), old)
        df = df.drop(columns=[pair.old_col, pair.new_col])
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recebe a coorte analítica (features BRUTAS + era + duration + event) e devolve o
    frame com todas as features derivadas/limpas, pronto para o split.
    """
    df = harmonize_and_combine_pairs(df)

    # --- Sentinelas (ADR-005) — dos códigos brutos de linfonodo e do tamanho combinado ---
    df["node_count"] = decode_node_count(df[COL_NODES_POSITIVE])
    df["node_status"] = decode_node_status(df[COL_NODES_POSITIVE])
    df["nodes_examined_n"] = decode_nodes_examined(df[COL_NODES_EXAMINED])
    df["tumor_size_diffuse"] = tumor_size_diffuse_flag(df["tumor_size_mm"])
    df["tumor_size_mm"] = decode_tumor_size(df["tumor_size_mm"])
    df = df.drop(columns=[COL_NODES_POSITIVE, COL_NODES_EXAMINED])

    # --- Features agrupadas ---
    df["age_mid"] = decode_age_mid(df[COL_AGE])
    df["surgery_group"] = group_surgery(df[COL_SURGERY])
    df["radiation_group"] = group_radiation(df[COL_RADIATION])
    df["stage"] = clean_stage(df[COL_STAGE])
    df = df.rename(columns={COL_RACE: "race", COL_BREAST_SUBTYPE: "breast_subtype",
                            COL_CHEMOTHERAPY: "chemotherapy"})
    df = df.drop(columns=[COL_AGE, COL_SURGERY, COL_RADIATION, COL_STAGE])

    # --- Mascarar tokens de nulo REAIS nas categóricas de texto ---
    for col in ("race", "er_status", "pr_status", "her2_status", "breast_subtype"):
        df[col] = _mask(df[col], REAL_NULL_TOKENS)
    df["tumor_grade"] = _mask(df["tumor_grade"], REAL_NULL_TOKENS | GRADE_NULL_TOKENS)

    keep = LABEL_STRUCTURAL_COLS + NUMERIC_FEATURES + CATEGORICAL_FEATURES + KEPT_NOT_MODELED + ["tumor_size_diffuse"]
    return df[keep]
