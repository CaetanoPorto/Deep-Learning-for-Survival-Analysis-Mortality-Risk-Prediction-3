"""Etapa 6 do RUNBOOK — baselines.

Ordem obrigatória: Cox -> teste de proporcionalidade -> Cox+splines -> RSF -> GBS.
Todos partem da MESMA matriz pré-processada e do MESMO split (fit só no treino).

Critério de aceite por modelo: 0,55 < C-index de teste < 0,95 e |gap treino-teste| < 0,05.
Referência da literatura: C-index 0,70-0,80 nesta base.

Uso:
    python scripts/run_baselines.py --sample-n 100000   # sanidade local (default)
    python scripts/run_baselines.py --full              # base inteira (Colab / demorado)
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
from src.config import RANDOM_SEED, UNO_TAU_PERCENTILE, set_global_seed
from src.data.build import build_analytic_frame
from src.data.loading import load_raw_full, load_raw_sample
from src.evaluate.metrics import harrell_cindex, to_structured, uno_cindex
from src.models.boosting import fit_gbs, gbs_risk
from src.models.cox import (
    build_cox_frame, build_spline_features, cox_risk, fit_cox, proportional_hazards_summary,
)
from src.models.forest import fit_rsf, rsf_risk
from src.preprocessing.split import train_val_test_split
from src.preprocessing.transform import fit_transform_splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-n", type=int, default=100_000)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)

    if args.full:
        print("Carregando base completa...")
        raw = load_raw_full()
    else:
        print(f"Carregando amostra de {args.sample_n} linhas (seed={RANDOM_SEED})...")
        raw = load_raw_sample(sample_n=args.sample_n, seed=RANDOM_SEED)

    df = build_analytic_frame(raw).analytic
    split = train_val_test_split(df)
    pre = fit_transform_splits(split.train, split.val, split.test)
    print(f"coorte {len(df):,} | train {len(split.train):,} | features {len(pre.feature_names)}")

    dur = {k: getattr(split, k)["duration"].to_numpy() for k in ("train", "val", "test")}
    ev = {k: getattr(split, k)["event"].to_numpy() for k in ("train", "val", "test")}
    y = {k: to_structured(dur[k], ev[k]) for k in ("train", "val", "test")}
    tau = float(np.percentile(dur["train"], UNO_TAU_PERCENTILE))

    results = {}  # nome -> (harrell_train, harrell_val, harrell_test, uno_test)

    def evaluate(name, risk_train, risk_val, risk_test):
        h = {s: harrell_cindex(dur[s], ev[s], r)
             for s, r in (("train", risk_train), ("val", risk_val), ("test", risk_test))}
        u_test = uno_cindex(y["train"], y["test"], risk_test, tau=tau)
        results[name] = (h["train"], h["val"], h["test"], u_test)
        print(f"  {name:14} Harrell tr/val/te = {h['train']:.4f}/{h['val']:.4f}/{h['test']:.4f}"
              f" | Uno teste = {u_test:.4f}")

    # --- 1. Cox ---
    print("\n[1] Cox proporcional (ridge)...")
    t0 = time.time()
    frames = {s: build_cox_frame(getattr(pre, f"X_{s}"), pre.feature_names, dur[s], ev[s])
              for s in ("train", "val", "test")}
    cox = fit_cox(frames["train"])
    evaluate("Cox", *(cox_risk(cox, frames[s]) for s in ("train", "val", "test")))
    print(f"  ({time.time() - t0:.0f}s)")

    # --- 2. Teste de proporcionalidade (Schoenfeld) ---
    print("\n[2] Teste de proporcionalidade (Schoenfeld)...")
    ph = proportional_hazards_summary(cox, frames["train"])
    print(f"  covariáveis: {ph.n_covariates} | violam p<0,05: {ph.n_violating_05} | p<0,01: {ph.n_violating_01}")
    print(f"  (N grande -> quase tudo rejeita; o que importa é o tamanho do desvio.)")
    print(f"  maiores estatísticas de teste:\n{ph.top[['test_statistic', 'p']].to_string()}")

    # --- 3. Cox + splines ---
    print("\n[3] Cox + splines nas contínuas...")
    t0 = time.time()
    Xtr_s, Xva_s, Xte_s, names_s = build_spline_features(pre.X_train, pre.X_val, pre.X_test, pre.feature_names)
    sframes = {s: build_cox_frame(X, names_s, dur[s], ev[s])
               for s, X in (("train", Xtr_s), ("val", Xva_s), ("test", Xte_s))}
    cox_s = fit_cox(sframes["train"])
    evaluate("Cox+splines", *(cox_risk(cox_s, sframes[s]) for s in ("train", "val", "test")))
    print(f"  ({time.time() - t0:.0f}s)")

    # --- 4. Random Survival Forest ---
    print("\n[4] Random Survival Forest...")
    t0 = time.time()
    rsf = fit_rsf(pre.X_train, y["train"])
    evaluate("RSF", *(rsf_risk(rsf, getattr(pre, f"X_{s}")) for s in ("train", "val", "test")))
    print(f"  ({time.time() - t0:.0f}s)")

    # --- 5. Gradient Boosting de sobrevivência ---
    print("\n[5] Gradient Boosting de sobrevivência...")
    t0 = time.time()
    gbs = fit_gbs(pre.X_train, y["train"])
    evaluate("GBS", *(gbs_risk(gbs, getattr(pre, f"X_{s}")) for s in ("train", "val", "test")))
    print(f"  ({time.time() - t0:.0f}s)")

    # --- Critério de aceite ---
    print("\n=== Etapa 6 — critério de aceite (Harrell de teste) ===")
    all_checks = []
    for name, (htr, _, hte, _) in results.items():
        all_checks += check_cindex(name, htr, hte)
    print(format_checks(all_checks))

    n_fail = sum(1 for c in all_checks if not c.ok)
    print("\n=== Resultado ===")
    print(f"checagens: {len(all_checks) - n_fail}/{len(all_checks)} OK")
    if all_passed(all_checks):
        print("\nCRITÉRIO DE ACEITE DA ETAPA 6: PASSOU [OK]")
        return 0
    print("\nCRITÉRIO DE ACEITE DA ETAPA 6: FALHOU [X] — PARE (ver Vazamento C-Index 1.0 / Armadilha da era).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
