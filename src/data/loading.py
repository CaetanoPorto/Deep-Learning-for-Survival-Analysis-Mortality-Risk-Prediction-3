"""Leitura do CSV bruto do SEER.

Tudo é lido como string (`dtype=str`, sem inferência de nulo do pandas): a limpeza de
"falsos nulos" e sentinelas em etapas posteriores precisa decidir ela mesma o que é
ausente. Deixar o pandas inferir cedo demais esconderia justamente os casos a auditar.

Além da leitura completa/amostrada, há um iterador em chunks — a Etapa 0 (perfil
empírico) varre 1,36M linhas contando valores sem materializar o CSV inteiro em RAM.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, RAW_CSV_PATH, READ_CSV_KWARGS


def read_header(path: Path = RAW_CSV_PATH) -> list[str]:
    """Lê apenas o cabeçalho (nomes de coluna, na ordem do arquivo)."""
    return list(pd.read_csv(path, nrows=0, **READ_CSV_KWARGS).columns)


def count_data_rows(path: Path = RAW_CSV_PATH) -> int:
    """Conta linhas de dados (exclui o header) por uma passada binária rápida — evita
    depender de um N hardcoded que ficaria defasado se o export do SEER*Stat mudar.
    """
    with open(path, "rb") as f:
        n_lines = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))
    return n_lines - 1  # desconta o header


def iter_raw_chunks(
    chunksize: int = 200_000, path: Path = RAW_CSV_PATH
) -> Iterator[pd.DataFrame]:
    """Itera o CSV em blocos de `chunksize` linhas (todas as colunas como string).

    Uso: perfilagem/validação de invariantes em streaming, sem carregar o arquivo
    inteiro. O consumidor agrega contagens por bloco.
    """
    yield from pd.read_csv(path, chunksize=chunksize, **READ_CSV_KWARGS)


def load_raw_full(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Carrega o CSV inteiro (~1,36M linhas) em memória. Usar na rodada final; para
    prototipar, prefira `load_raw_sample`.
    """
    return pd.read_csv(path, **READ_CSV_KWARGS)


def load_raw_sample(
    sample_n: int, seed: int = RANDOM_SEED, path: Path = RAW_CSV_PATH
) -> pd.DataFrame:
    """Amostra `sample_n` linhas do CSV sem carregar o arquivo inteiro.

    Usa `skiprows` com índices sorteados: o parser C ainda percorre o arquivo, mas
    nunca materializa as linhas não sorteadas — permite prototipar em segundos e poucos
    MB de RAM antes de escalar para o dataset completo.
    """
    n_total = count_data_rows(path)
    sample_n = min(sample_n, n_total)
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(n_total, size=sample_n, replace=False).tolist())

    def _skip(row_idx: int) -> bool:
        if row_idx == 0:  # header
            return False
        return (row_idx - 1) not in chosen

    return pd.read_csv(path, skiprows=_skip, **READ_CSV_KWARGS)
