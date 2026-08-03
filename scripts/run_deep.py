"""Etapa 7 do RUNBOOK — modelos profundos (DeepSurv, DeepHit).

Fluxo por modelo: LR finder -> treino com early stopping (patience 10) -> C-index. Roda
>= 3 sementes e reporta média ± desvio. DeepHit usa 20 bins por quantis (com bin p/ t=0).

Local: `--smoke` (subamostra pequena, 1 semente, poucas épocas) só para validar o
pipeline em CPU. O treino de verdade (base inteira, mais sementes) roda no Colab (GPU) —
ver notebooks/colab_deep.ipynb.

Uso:
    python scripts/run_deep.py --smoke
    python scripts/run_deep.py --sample-n 100000 --seeds 3        # comparação (ADR-010)
    python scripts/run_deep.py --full --seeds 5                   # Colab
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from src.checks.invariants import all_passed, check_cindex, format_checks
from src.config import (
    COMPARISON_SAMPLE_N, MIN_SEEDS, TrainConfig, UNO_TAU_PERCENTILE, set_global_seed,
)
from src.data.build import build_analytic_frame
from src.data.loading import load_raw_full, load_raw_sample
from src.evaluate.curves import antolini_cindex
from src.evaluate.metrics import harrell_cindex, to_structured, uno_cindex
from src.models import deephit, deepsurv
from src.train.lr_finder import find_lr
from src.train.trainer import fit_with_early_stopping


def _prep(sample_n, full, smoke):
    if smoke:
        raw = load_raw_sample(20_000, seed=42)
    elif full:
        raw = load_raw_full()
    else:
        raw = load_raw_sample(sample_n, seed=42)
    from src.preprocessing.split import train_val_test_split
    from src.preprocessing.transform import fit_transform_splits
    df = build_analytic_frame(raw).analytic
    split = train_val_test_split(df)
    pre = fit_transform_splits(split.train, split.val, split.test)
    dur = {k: getattr(split, k)["duration"].to_numpy() for k in ("train", "val", "test")}
    ev = {k: getattr(split, k)["event"].to_numpy() for k in ("train", "val", "test")}
    X = {k: getattr(pre, f"X_{k}") for k in ("train", "val", "test")}
    return X, dur, ev, pre.feature_names


def _train_deepsurv(X, dur, ev, seed, config, eval_key):
    set_global_seed(seed)
    y_tr = deepsurv.make_target(dur["train"], ev["train"])
    y_va = deepsurv.make_target(dur["val"], ev["val"])
    in_f = X["train"].shape[1]
    lr = find_lr(deepsurv.build_model(deepsurv.build_net(in_f)), X["train"], y_tr, config)
    model = deepsurv.build_model(deepsurv.build_net(in_f))  # rede fresca após o LR finder
    fit_with_early_stopping(model, X["train"], y_tr, X["val"], y_va, config, learning_rate=lr)
    deepsurv.compute_baseline_hazards(model, X["train"], y_tr)
    risk = deepsurv.predict_risk(model, X[eval_key])
    surv = deepsurv.predict_survival_function(model, X[eval_key])
    risk_tr = deepsurv.predict_risk(model, X["train"])
    return risk_tr, risk, surv


def _train_deephit(X, dur, ev, seed, config, eval_key):
    set_global_seed(seed)
    labtrans = deephit.fit_label_transform(deephit.build_label_transform(), dur["train"], ev["train"])
    y_tr = deephit.make_target(labtrans, dur["train"], ev["train"])
    y_va = deephit.make_target(labtrans, dur["val"], ev["val"])
    in_f = X["train"].shape[1]
    lr = find_lr(deephit.build_model(deephit.build_net(in_f, labtrans), labtrans), X["train"], y_tr, config)
    net = deephit.build_net(in_f, labtrans)
    model = deephit.build_model(net, labtrans)
    fit_with_early_stopping(model, X["train"], y_tr, X["val"], y_va, config, learning_rate=lr)
    risk = deephit.predict_risk(model, X[eval_key])
    surv = deephit.predict_survival_function(model, X[eval_key])
    risk_tr = deephit.predict_risk(model, X["train"])
    return risk_tr, risk, surv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-n", type=int, default=COMPARISON_SAMPLE_N)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, default=MIN_SEEDS)
    parser.add_argument("--epochs", type=int, default=None)
    # Avalia no VAL por padrão: robustez/tuning dos modelos profundos SEM tocar no teste.
    # O teste é aberto uma única vez, só em run_evaluation.py --eval-set test.
    parser.add_argument("--eval-set", choices=["val", "test"], default="val")
    args = parser.parse_args()

    config = TrainConfig()
    if args.epochs:
        config.epochs = args.epochs
    seeds = list(range(1, (1 if args.smoke else args.seeds) + 1))
    if args.smoke:
        config.epochs = min(config.epochs, 30)

    eval_key = "test" if args.smoke else args.eval_set  # smoke pode usar test (é amostra)
    print(f"Preparando dados (smoke={args.smoke}, full={args.full}, sample={args.sample_n}, aval={eval_key})...")
    X, dur, ev, names = _prep(args.sample_n, args.full, args.smoke)
    y = {k: to_structured(dur[k], ev[k]) for k in ("train", "val", "test")}
    tau = float(np.percentile(dur["train"], UNO_TAU_PERCENTILE))
    print(f"train {len(X['train']):,} | avaliação ({eval_key}) {len(X[eval_key]):,} | features {X['train'].shape[1]} | sementes {seeds}")

    scores = {"DeepSurv": [], "DeepHit": []}  # cada item: (harrell_tr, harrell_ev, uno_ev, antolini_ev)
    for seed in seeds:
        print(f"\n--- semente {seed} ---")
        for name, trainer in (("DeepSurv", _train_deepsurv), ("DeepHit", _train_deephit)):
            t0 = time.time()
            risk_tr, risk_ev, surv_ev = trainer(X, dur, ev, seed, config, eval_key)
            h_tr = harrell_cindex(dur["train"], ev["train"], risk_tr)
            h_ev = harrell_cindex(dur[eval_key], ev[eval_key], risk_ev)
            u_ev = uno_cindex(y["train"], y[eval_key], risk_ev, tau=tau)
            a_ev = antolini_cindex(surv_ev, dur[eval_key], ev[eval_key])
            scores[name].append((h_tr, h_ev, u_ev, a_ev))
            print(f"  {name:9} Harrell {eval_key}={h_ev:.4f} | Uno={u_ev:.4f} | Antolini={a_ev:.4f}  ({time.time()-t0:.0f}s)")

    print(f"\n=== Etapa 7 — resumo em {eval_key} (média ± desvio entre sementes) ===")
    checks = []
    for name, rows in scores.items():
        arr = np.array(rows)
        h_tr, h_ev, u_ev, a_ev = arr.mean(axis=0)
        s_ev = arr.std(axis=0)
        print(f"  {name:9} Harrell={h_ev:.4f}±{s_ev[1]:.4f} | Uno={u_ev:.4f} | Antolini={a_ev:.4f}±{s_ev[3]:.4f}")
        checks += check_cindex(name, h_tr, h_ev)

    print("\n=== Critério de aceite (0,55<C<0,95; gap<0,05) ===")
    print(format_checks(checks))
    if all_passed(checks):
        print("\nETAPA 7 (smoke/local): PASSOU [OK]" if args.smoke else "\nETAPA 7: PASSOU [OK]")
        return 0
    print("\nETAPA 7: FALHOU [X] — PARE.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
