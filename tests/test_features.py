"""Testes das features agrupadas (idade, cirurgia, radiação, estadiamento)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    RADIATION_BEAM,
    RADIATION_GROUP_MAP,
    RADIATION_IMPLANT,
    RADIATION_NONE_UNKNOWN,
    RADIATION_REFUSED,
    SURGERY_CONSERVATIVE,
    SURGERY_MASTECTOMY,
    SURGERY_NONE,
    SURGERY_OTHER_UNKNOWN,
)
from src.data.features import clean_stage, decode_age_mid, group_radiation, group_surgery, unmapped_age


def test_age_mid_pontos_medios_e_extremos():
    s = pd.Series(["00 years", "01-04 years", "60-64 years", "90+ years"])
    assert list(decode_age_mid(s)) == [0.0, 2.5, 62.0, 92.0]


def test_unmapped_age_detecta_categoria_nova():
    assert unmapped_age(pd.Series(["60-64 years", "999 years"])) == ["999 years"]


def test_group_surgery_familias():
    s = pd.Series(["00", "22", "20", "24", "41", "51", "59", "40", "99", "30", "10", "90"])
    out = list(group_surgery(s))
    assert out == [
        SURGERY_NONE,          # 00
        SURGERY_CONSERVATIVE, SURGERY_CONSERVATIVE, SURGERY_CONSERVATIVE,  # 20-24
        SURGERY_MASTECTOMY, SURGERY_MASTECTOMY, SURGERY_MASTECTOMY, SURGERY_MASTECTOMY,  # 40-59
        SURGERY_OTHER_UNKNOWN, SURGERY_OTHER_UNKNOWN, SURGERY_OTHER_UNKNOWN, SURGERY_OTHER_UNKNOWN,  # 99,30,10,90
    ]


def test_group_radiation_cobre_os_8_valores():
    s = pd.Series(list(RADIATION_GROUP_MAP.keys()))
    out = group_radiation(s)
    assert out.notna().all()  # cobertura total, nenhum NaN
    # amostras pontuais dos 4 grupos
    m = dict(zip(RADIATION_GROUP_MAP.keys(), out))
    assert m["Beam radiation"] == RADIATION_BEAM
    assert m["Radiation, NOS  method or source not specified"] == RADIATION_BEAM
    assert m["Radioisotopes (1988+)"] == RADIATION_IMPLANT
    assert m["Refused (1988+)"] == RADIATION_REFUSED
    assert m["None/Unknown"] == RADIATION_NONE_UNKNOWN


def test_clean_stage_ordinal():
    s = pd.Series([
        "In situ", "Localized only", "Regional lymph nodes involved only",
        "Regional by BOTH direct extension and lymph node involvement",
        "Distant site(s)/node(s) involved", "Blank(s)", "Unknown/unstaged/unspecified/DCO",
    ])
    out = clean_stage(s)
    assert list(out[:5]) == ["in_situ", "localized", "regional", "regional", "distant"]
    assert out[5:].isna().all()  # Blank(s) e Unknown -> NaN
    assert out.cat.ordered
