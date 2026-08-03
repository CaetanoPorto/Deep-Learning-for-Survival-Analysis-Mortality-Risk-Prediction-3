"""DeepHit: MLP + tempo discreto, via `pycox.models.DeepHitSingle`.

Prevê uma distribuição sobre uma grade discreta de tempo — não assume risco
proporcional (por isso é a resposta à violação de proporcionalidade vista no Schoenfeld,
Etapa 6). A discretização do ALVO é aprendida SÓ no treino (`fit`) e aplicada
(`transform`) em val/teste, seguindo a regra anti-vazamento; os quantis incluem um bin
para t=0 (há muitos eventos/censuras no mês 0 — ADR-004).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torchtuples as tt
from pycox.models import DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

from src.config import DeepHitConfig, MLPConfig


def build_label_transform(config: DeepHitConfig = DeepHitConfig()) -> LabTransDiscreteTime:
    return DeepHitSingle.label_transform(config.num_time_bins, scheme=config.scheme)


def fit_label_transform(labtrans: LabTransDiscreteTime, duration_train, event_train) -> LabTransDiscreteTime:
    """Aprende os cortes de tempo (`cuts`) SÓ a partir do treino."""
    labtrans.fit(np.asarray(duration_train, dtype="float32"), np.asarray(event_train))
    return labtrans


def make_target(labtrans: LabTransDiscreteTime, duration, event) -> tuple[np.ndarray, np.ndarray]:
    """Discretiza (duration, event) com os cortes já aprendidos (transform, nunca
    fit_transform fora do treino).
    """
    return labtrans.transform(np.asarray(duration, dtype="float32"), np.asarray(event))


def build_net(in_features: int, labtrans: LabTransDiscreteTime, mlp_config: MLPConfig = MLPConfig()) -> tt.practical.MLPVanilla:
    num_nodes = [mlp_config.hidden_units] * mlp_config.num_layers
    return tt.practical.MLPVanilla(
        in_features, num_nodes, out_features=labtrans.out_features,
        batch_norm=mlp_config.batch_norm, dropout=mlp_config.dropout,
    )


def build_model(net: tt.practical.MLPVanilla, labtrans: LabTransDiscreteTime, config: DeepHitConfig = DeepHitConfig()) -> DeepHitSingle:
    return DeepHitSingle(net, tt.optim.Adam, alpha=config.alpha, sigma=config.sigma, duration_index=labtrans.cuts)


def predict_survival_function(model: DeepHitSingle, x: np.ndarray) -> pd.DataFrame:
    return model.predict_surv_df(x)


def predict_risk(model: DeepHitSingle, x: np.ndarray) -> np.ndarray:
    """Escore de risco = soma de S(t) negada (área sob a curva ~ tempo médio restrito).

    NÃO usar `1 - S(t_max)`: o último bin colapsa para ~0 para quase todo mundo (artefato
    de tempo discreto) e o ranking sai invertido. A soma usa a curva inteira; deu C-index
    consistente com Cox/DeepSurv. É provisório para sanidade — a avaliação final do
    DeepHit usa o C-index de Antolini (tempo-dependente).
    """
    surv = model.predict_surv_df(x)
    return -surv.sum(axis=0).to_numpy()
