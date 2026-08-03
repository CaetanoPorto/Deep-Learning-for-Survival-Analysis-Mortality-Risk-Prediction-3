"""Loop de treino com early stopping — comum a DeepSurv e DeepHit (ambos herdam
`torchtuples.Model`, então batching/early stopping/logging funcionam igual).
"""

from __future__ import annotations

import numpy as np
import torchtuples as tt

from src.config import TrainConfig


def fit_with_early_stopping(
    model: tt.Model,
    x_train: np.ndarray,
    y_train: tuple,
    x_val: np.ndarray,
    y_val: tuple,
    config: TrainConfig = TrainConfig(),
    learning_rate: float | None = None,
    verbose: bool = False,
):
    """Treina `model` com early stopping na perda de validação (patience do config).
    Devolve o log (histórico de perda por época) para inspeção/gráfico da banca.
    """
    if learning_rate is not None:
        model.optimizer.set_lr(learning_rate)
    callbacks = [tt.callbacks.EarlyStopping(patience=config.patience)]
    return model.fit(
        x_train, y_train,
        batch_size=config.batch_size, epochs=config.epochs,
        callbacks=callbacks, verbose=verbose, val_data=(x_val, y_val),
    )
