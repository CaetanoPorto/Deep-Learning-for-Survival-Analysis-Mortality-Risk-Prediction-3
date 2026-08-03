"""Testes da reconstrução de era (a lógica mais crítica do pipeline).

A validação contra o CSV real está em scripts/profile_dataset.py (Etapa 0); aqui
exercitamos as 5 ramificações e a ordem de prioridade em fixtures sintéticas pequenas,
sem depender do CSV.
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    BLANK_TOKEN,
    BREAST_SUBTYPE_NA_TOKEN,
    COL_BREAST_SUBTYPE,
    COL_GRADE_2018,
    COL_STAGE,
    COL_TUMOR_SIZE_SUMMARY,
    ERA_2000_2003,
    ERA_2004_2009,
    ERA_2010_2015,
    ERA_2016_2017,
    ERA_2018_2022,
    ERA_LABELS,
    RAW_COLUMNS,
)
from src.data.era import ERA_COLUMN, reconstruct_era


def _row(stage: str, grade2018: str, size2016: str, subtype: str) -> dict:
    return {
        COL_STAGE: stage,
        COL_GRADE_2018: grade2018,
        COL_TUMOR_SIZE_SUMMARY: size2016,
        COL_BREAST_SUBTYPE: subtype,
    }


def test_cada_ramificacao_da_era():
    df = pd.DataFrame(
        [
            # stage == Blank(s)                                        -> 2000-2003
            _row(BLANK_TOKEN, BLANK_TOKEN, BLANK_TOKEN, BREAST_SUBTYPE_NA_TOKEN),
            # grade 2018 preenchido                                   -> 2018-2022
            _row("Localized only", "2", "18", "HR+/HER2-"),
            # tumor size summary preenchido                           -> 2016-2017
            _row("Localized only", BLANK_TOKEN, "18", "HR+/HER2-"),
            # subtype != "Recode not available"                       -> 2010-2015
            _row("Localized only", BLANK_TOKEN, BLANK_TOKEN, "HR+/HER2-"),
            # tudo estrutural / subtype não disponível                -> 2004-2009
            _row("Localized only", BLANK_TOKEN, BLANK_TOKEN, BREAST_SUBTYPE_NA_TOKEN),
        ]
    )
    era = reconstruct_era(df)
    assert list(era) == [ERA_2000_2003, ERA_2018_2022, ERA_2016_2017, ERA_2010_2015, ERA_2004_2009]


def test_prioridade_stage_vence_tudo():
    # Mesmo com grade 2018 preenchido, stage == Blank(s) tem prioridade -> 2000-2003.
    df = pd.DataFrame([_row(BLANK_TOKEN, "3", "20", "HR-/HER2+")])
    assert list(reconstruct_era(df)) == [ERA_2000_2003]


def test_subtype_unknown_conta_como_2010_2015():
    # "Unknown" != "Recode not available": subtipo existe (era >= 2010), então cai em
    # 2010-2015 (2018 e 2016-17 já teriam sido capturados antes por prioridade).
    df = pd.DataFrame([_row("Localized only", BLANK_TOKEN, BLANK_TOKEN, "Unknown")])
    assert list(reconstruct_era(df)) == [ERA_2010_2015]


def test_dtype_categorico_com_5_niveis():
    df = pd.DataFrame([_row(BLANK_TOKEN, BLANK_TOKEN, BLANK_TOKEN, BREAST_SUBTYPE_NA_TOKEN)])
    era = reconstruct_era(df)
    assert era.name == ERA_COLUMN
    assert isinstance(era.dtype, pd.CategoricalDtype)
    assert list(era.cat.categories) == ERA_LABELS


def test_schema_tem_22_colunas_unicas():
    assert len(RAW_COLUMNS) == 22
    assert len(set(RAW_COLUMNS)) == 22
