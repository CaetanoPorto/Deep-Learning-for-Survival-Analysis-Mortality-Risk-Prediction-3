"""DeepSurv: MLP + verossimilhança parcial de Cox, via `pycox.models.CoxPH`.

Usa o `MLPVanilla` do torchtuples (a rede de referência do pycox), para não reintroduzir
um detalhe de implementação (init de pesos, ordem de batch-norm/dropout) que mudaria a
comparação com a literatura sem necessidade. Hazard proporcional, log-risco não linear.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torchtuples as tt
from pycox.models import CoxPH

from src.config import MLPConfig


def build_net(in_features: int, mlp_config: MLPConfig = MLPConfig()) -> tt.practical.MLPVanilla:
    num_nodes = [mlp_config.hidden_units] * mlp_config.num_layers
    return tt.practical.MLPVanilla(
        in_features, num_nodes, out_features=1,
        batch_norm=mlp_config.batch_norm, dropout=mlp_config.dropout,
        output_bias=False,  # convenção DeepSurv: só a ordem entre escores importa
    )


def build_model(net: tt.practical.MLPVanilla) -> CoxPH:
    return CoxPH(net, tt.optim.Adam)


def make_target(duration, event) -> tuple[np.ndarray, np.ndarray]:
    """pycox espera (durations, events) como float32."""
    return np.asarray(duration, dtype="float32"), np.asarray(event, dtype="float32")


def predict_risk(model: CoxPH, x: np.ndarray) -> np.ndarray:
    """Log relative hazard: maior = maior risco (convenção de evaluate.metrics)."""
    return model.predict(x).flatten()


def compute_baseline_hazards(model: CoxPH, x_train: np.ndarray, y_train: tuple) -> None:
    """Estima a hazard baseline (Breslow) A PARTIR DO TREINO — obrigatório antes de
    `predict_survival_function`, e nunca com val/teste (senão a curva de um paciente do
    teste seria informada por outros do próprio teste: vazamento).
    """
    model.compute_baseline_hazards(input=x_train, target=y_train)


def predict_survival_function(model: CoxPH, x: np.ndarray) -> pd.DataFrame:
    """Curva S(t) por indivíduo (requer compute_baseline_hazards já chamado)."""
    return model.predict_surv_df(x)
