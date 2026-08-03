"""Testes da lógica de coorte (Etapa 1) em fixtures sintéticas.

A validação contra o CSV real está em scripts/run_cohort.py; aqui cobrimos a mecânica:
homens saem, tempo desconhecido é separado (não descartado), duration==0 fica.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import (
    BLANK_TOKEN,
    BREAST_SUBTYPE_NA_TOKEN,
    COL_BREAST_SUBTYPE,
    COL_GRADE_2018,
    COL_SEX,
    COL_STAGE,
    COL_SURVIVAL_MONTHS,
    COL_TUMOR_SIZE_SUMMARY,
    COL_VITAL_STATUS,
    ERA_2000_2003,
    ERA_2010_2015,
    ERA_2018_2022,
    ERA_LABELS,
)
from src.data.build import build_cohort
from src.data.cohort import apply_cohort, restrict_to_eras
from src.data.era import ERA_COLUMN


def _cohort_df() -> pd.DataFrame:
    rows = [
        # sexo,   survival, vital,   era
        ("Female", "0012", "Dead", ERA_2010_2015),   # analítica
        ("Female", "0000", "Alive", ERA_2018_2022),  # analítica (duration==0 mantido)
        ("Female", "Unknown", "Dead", ERA_2000_2003),  # separada (tempo desconhecido)
        ("Male", "0050", "Dead", ERA_2010_2015),      # removido (homem)
        ("Male", "Unknown", "Dead", ERA_2018_2022),   # removido (homem + tempo desc.)
        ("Female", "0100", "Alive", ERA_2010_2015),   # analítica
    ]
    df = pd.DataFrame(rows, columns=[COL_SEX, COL_SURVIVAL_MONTHS, COL_VITAL_STATUS, "era_raw"])
    df[ERA_COLUMN] = pd.Categorical(df["era_raw"], categories=ERA_LABELS, ordered=True)
    return df.drop(columns=["era_raw"])


def test_fluxo_de_coorte_conta_certo():
    result = apply_cohort(_cohort_df())
    flow = result.flow
    assert flow.n_raw == 6
    assert flow.n_male_removed == 2
    assert flow.n_female == 4
    assert flow.n_unknown_time_total == 2
    assert flow.n_unknown_time_male == 1
    assert flow.n_unknown_time_female_removed == 1
    assert flow.n_analytic == 3
    assert flow.n_unknown_time_female_dead == 1  # o tempo desconhecido feminino é óbito
    assert flow.n_duration_zero_analytic == 1


def test_tempo_desconhecido_e_separado_nao_descartado():
    result = apply_cohort(_cohort_df())
    # a linha de tempo desconhecido não some: fica reservada para sensibilidade.
    assert len(result.unknown_time) == 1
    assert (result.unknown_time[COL_SURVIVAL_MONTHS] == "Unknown").all()
    # e não contamina a coorte analítica.
    assert (result.analytic[COL_SURVIVAL_MONTHS] != "Unknown").all()
    assert (result.analytic[COL_SEX] == "Female").all()


def test_apply_cohort_exige_era_reconstruida():
    df = _cohort_df().drop(columns=[ERA_COLUMN])
    with pytest.raises(KeyError):
        apply_cohort(df)


def test_build_cohort_reconstroi_era_antes_do_filtro():
    # raw sintético com as 4 colunas de era + sexo/desfecho; build_cohort deve
    # reconstruir a era, filtrar e montar o alvo, tudo na ordem certa.
    raw = pd.DataFrame(
        [
            # 2000-2003 (stage Blank), feminina, tempo conhecido
            (BLANK_TOKEN, BLANK_TOKEN, BLANK_TOKEN, BREAST_SUBTYPE_NA_TOKEN, "Female", "0030", "Dead"),
            # 2018-2022 (grade 2018 preenchido), feminina, tempo conhecido
            ("Localized only", "2", "18", "HR+/HER2-", "Female", "0005", "Alive"),
            # homem — deve sair
            (BLANK_TOKEN, BLANK_TOKEN, BLANK_TOKEN, BREAST_SUBTYPE_NA_TOKEN, "Male", "0040", "Dead"),
        ],
        columns=[
            COL_STAGE, COL_GRADE_2018, COL_TUMOR_SIZE_SUMMARY, COL_BREAST_SUBTYPE,
            COL_SEX, COL_SURVIVAL_MONTHS, COL_VITAL_STATUS,
        ],
    )
    result = build_cohort(raw)
    assert result.flow.n_analytic == 2
    assert result.flow.n_male_removed == 1
    assert ERA_COLUMN in result.analytic.columns
    assert list(result.analytic[ERA_COLUMN]) == [ERA_2000_2003, ERA_2018_2022]
    # alvo montado e rótulos brutos removidos
    assert {"duration", "event"} <= set(result.analytic.columns)
    assert COL_SURVIVAL_MONTHS not in result.analytic.columns


def test_restrict_none_devolve_base_completa():
    df = _cohort_df()
    out = restrict_to_eras(df, None)
    assert len(out) == len(df)


def test_restrict_filtra_eras_selecionadas():
    df = _cohort_df()
    out = restrict_to_eras(df, [ERA_2010_2015])
    assert set(out[ERA_COLUMN]) == {ERA_2010_2015}
    assert len(out) == int((df[ERA_COLUMN] == ERA_2010_2015).sum())
