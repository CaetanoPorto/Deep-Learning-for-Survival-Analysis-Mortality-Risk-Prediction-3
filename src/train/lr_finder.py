"""LR finder (teste de faixa de learning rate) antes de fixar a taxa — Plano de Modelagem.

Roda um mini-treino aumentando o LR e observa onde a perda começa a divergir; sugere um
LR na região estável, em vez de chutar.
"""

from __future__ import annotations

import numpy as np
import torchtuples as tt

from src.config import TrainConfig


def find_lr(
    model: tt.Model, x_train: np.ndarray, y_train: tuple, config: TrainConfig = TrainConfig()
) -> float:
    """Devolve o LR sugerido pelo LR finder do torchtuples. Se algo falhar (ex.: amostra
    pequena no smoke), cai no default do config.
    """
    try:
        finder = model.lr_finder(x_train, y_train, batch_size=config.batch_size, tolerance=10)
        best = finder.get_best_lr()
        # limita a uma faixa sã (o get_best_lr às vezes devolve valores extremos).
        return float(np.clip(best, 1e-4, 1e-1))
    except Exception:
        return config.learning_rate
