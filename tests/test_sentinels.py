"""Testes da decodificação de sentinelas (ADR-005) — os casos-limite que uma amostra
aleatória pode não conter (95/97/98/99, faixas 991-997, teto de 200 mm)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    NODE_STATUS_NEGATIVE,
    NODE_STATUS_NOT_ASSESSED,
    NODE_STATUS_POSITIVE,
    NODE_STATUS_UNKNOWN,
)
from src.data.sentinels import (
    decode_node_count,
    decode_node_status,
    decode_nodes_examined,
    decode_tumor_size,
    tumor_size_diffuse_flag,
)


def test_node_count_mascara_codigos():
    s = pd.Series(["00", "05", "90", "95", "97", "98", "99", "01"])
    out = decode_node_count(s)
    assert list(out[:3]) == [0.0, 5.0, 90.0]
    assert out[3:7].isna().all()  # 95/97/98/99 -> NaN
    assert out.iloc[7] == 1.0


def test_node_status_recupera_positivos():
    s = pd.Series(["00", "01", "90", "95", "97", "98", "99"])
    out = list(decode_node_status(s))
    assert out == [
        NODE_STATUS_NEGATIVE,   # 00
        NODE_STATUS_POSITIVE,   # 01
        NODE_STATUS_POSITIVE,   # 90
        NODE_STATUS_POSITIVE,   # 95 (aspiração positiva)
        NODE_STATUS_POSITIVE,   # 97 (positivos, nº não especificado)
        NODE_STATUS_NOT_ASSESSED,  # 98 (nenhum examinado)
        NODE_STATUS_UNKNOWN,    # 99
    ]


def test_nodes_examined_mascara_96():
    s = pd.Series(["00", "90", "95", "96", "97", "98", "99"])
    out = decode_nodes_examined(s)
    assert list(out[:2]) == [0.0, 90.0]
    assert out[2:].isna().all()


def test_tumor_size_faixas_e_teto():
    s = pd.Series(["000", "001", "150", "200", "201", "400", "990",
                   "991", "992", "993", "994", "995", "996", "997", "998", "999"])
    out = decode_tumor_size(s)
    esperado = [0, 1, 150, 200, np.nan, np.nan, np.nan,
                5, 15, 25, 35, 45, np.nan, np.nan, np.nan, np.nan]
    for got, exp in zip(out, esperado):
        if np.isnan(exp):
            assert np.isnan(got)
        else:
            assert got == exp


def test_tumor_size_diffuse_flag():
    s = pd.Series(["998", "150", "998", "000"])
    assert list(tumor_size_diffuse_flag(s)) == [True, False, True, False]
