"""Etapa 8 do RUNBOOK — avaliação final.

Treina todos os modelos e produz: tabela comparativa (C-Harrell/Uno/Antolini/IBS/Brier +
IC bootstrap), curva de calibração por decil, métricas por era, validação temporal
(2010-15 -> 2016-17) e sensibilidade do tempo desconhecido.

>>> O CONJUNTO DE TESTE É ABERTO UMA ÚNICA VEZ (Protocolo de Validação). <<<
Por isso o default é `--eval-set val` (dry-run: valida todo o pipeline no conjunto de
VALIDAÇÃO). Só passe `--eval-set test` na rodada final e definitiva.

Uso:
    python scripts/run_evaluation.py --sample-n 15000            # dry-run no val (default)
    python scripts/run_evaluation.py --full --eval-set test --n-boot 200   # FINAL (uma vez)
"""

from __future__ import annotations
from src.train.trainer import fit_with_early_stopping
from src.preprocessing.transform import INPUT_COLUMNS, fit_transform_splits
from src.preprocessing.split import train_val_test_split
from src.models.forest import fit_rsf, rsf_risk, rsf_survival
from src.models.cox import (
    build_cox_frame, build_spline_features, cox_risk, cox_survival, fit_cox,
)
from src.models.boosting import fit_gbs, gbs_risk, gbs_survival
from src.models import deephit, deepsurv
from src.evaluate.temporal import temporal_validation_cox
from src.evaluate.stratified import metrics_by_era
from src.evaluate.sensitivity import unknown_time_sensitivity
from src.evaluate.metrics import to_structured
from src.evaluate.comparison import ModelPrediction, comparison_table
from src.evaluate.calibration import calibration_by_decile
from src.data.loading import load_raw_full, load_raw_sample
from src.data.era import ERA_COLUMN
from src.data.cleaning import clean_features
from src.data.build import build_analytic_frame
from src.config import (
    CALIBRATION_HORIZON, ERA_MAIN_COHORT_RECOMMENDED, RANDOM_SEED, TrainConfig,
    UNO_TAU_PERCENTILE, set_global_seed,
)
import numpy as np

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _train_all(pre, split, eval_key, config, include_gbs):
    """Treina os modelos e devolve {nome: (risk_eval, surv_eval, risk_train)} + o Cox
    (para calibração/sensibilidade/temporal, que precisam prever em X novos)."""
    names = pre.feature_names
    X = {k: getattr(pre, f"X_{k}") for k in ("train", "val", "test")}
    dur = {k: getattr(split, k)["duration"].to_numpy()
           for k in ("train", "val", "test")}
    ev = {k: getattr(split, k)["event"].to_numpy()
          for k in ("train", "val", "test")}
    preds, cox_model, cox_names = {}, None, None

    # Cox
    fr_tr = build_cox_frame(X["train"], names, dur["train"], ev["train"])
    fr_ev = build_cox_frame(X[eval_key], names, dur[eval_key], ev[eval_key])
    cox = fit_cox(fr_tr)
    preds["Cox"] = (cox_risk(cox, fr_ev), cox_survival(
        cox, fr_ev), cox_risk(cox, fr_tr))
    cox_model, cox_names = cox, names

    # Cox + splines
    xtr_s, xev_s, _, names_s = build_spline_features(
        X["train"], X[eval_key], X["train"], names)
    sfr_tr = build_cox_frame(xtr_s, names_s, dur["train"], ev["train"])
    sfr_ev = build_cox_frame(xev_s, names_s, dur[eval_key], ev[eval_key])
    cox_s = fit_cox(sfr_tr)
    preds["Cox+splines"] = (cox_risk(cox_s, sfr_ev),
                            cox_survival(cox_s, sfr_ev), cox_risk(cox_s, sfr_tr))

    # RSF
    y_tr = to_structured(dur["train"], ev["train"])
    rsf = fit_rsf(X["train"], y_tr)
    preds["RSF"] = (rsf_risk(rsf, X[eval_key]), rsf_survival(
        rsf, X[eval_key]), rsf_risk(rsf, X["train"]))

    # GBS (só se couber — ADR-010)
    if include_gbs:
        gbs = fit_gbs(X["train"], y_tr)
        preds["GBS"] = (gbs_risk(gbs, X[eval_key]), gbs_survival(
            gbs, X[eval_key]), gbs_risk(gbs, X["train"]))

    # DeepSurv
    set_global_seed(1)
    yds_tr, yds_va = deepsurv.make_target(
        dur["train"], ev["train"]), deepsurv.make_target(dur["val"], ev["val"])
    ds = deepsurv.build_model(deepsurv.build_net(X["train"].shape[1]))
    fit_with_early_stopping(ds, X["train"], yds_tr, X["val"], yds_va, config)
    deepsurv.compute_baseline_hazards(ds, X["train"], yds_tr)
    preds["DeepSurv"] = (deepsurv.predict_risk(ds, X[eval_key]), deepsurv.predict_survival_function(ds, X[eval_key]),
                         deepsurv.predict_risk(ds, X["train"]))

    # DeepHit
    set_global_seed(1)
    lt = deephit.fit_label_transform(
        deephit.build_label_transform(), dur["train"], ev["train"])
    ydh_tr, ydh_va = deephit.make_target(
        lt, dur["train"], ev["train"]), deephit.make_target(lt, dur["val"], ev["val"])
    dh = deephit.build_model(deephit.build_net(X["train"].shape[1], lt), lt)
    fit_with_early_stopping(dh, X["train"], ydh_tr, X["val"], ydh_va, config)
    preds["DeepHit"] = (deephit.predict_risk(dh, X[eval_key]), deephit.predict_survival_function(dh, X[eval_key]),
                        deephit.predict_risk(dh, X["train"]))

    return preds, (cox_model, cox_names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", choices=["val", "test"], default="val")
    parser.add_argument("--sample-n", type=int, default=15_000)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--no-gbs", action="store_true",
                        help="pular o GBS (não escala; ADR-010)")
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)

    if args.eval_set == "test":
        print("=" * 70)
        print("!!! ABRINDO O CONJUNTO DE TESTE — isto deve acontecer UMA ÚNICA VEZ. !!!")
        print("=" * 70)
    else:
        print("[dry-run] avaliando no conjunto de VALIDAÇÃO (o teste NÃO é tocado).")

    raw = load_raw_full() if args.full else load_raw_sample(args.sample_n, seed=42)
    result = build_analytic_frame(raw)
    df, unknown = result.analytic, result.unknown_time
    split = train_val_test_split(df)
    pre = fit_transform_splits(split.train, split.val, split.test)

    config = TrainConfig()
    config.epochs = 30 if (
        not args.full and args.eval_set == "val") else config.epochs
    eval_df = getattr(split, args.eval_set)
    dur_e, ev_e, era_e = eval_df["duration"].to_numpy(
    ), eval_df["event"].to_numpy(), eval_df[ERA_COLUMN].to_numpy()
    y_train = to_structured(split.train["duration"], split.train["event"])
    tau = float(np.percentile(split.train["duration"], UNO_TAU_PERCENTILE))

    include_gbs = not args.no_gbs
    if args.full and include_gbs:
        print(
            "[ADR-010] --full: GBS pulado (O(n²), inviável em ~944k; ele entra só na amostra fixa).")
        include_gbs = False

    print(f"\ntreino {len(split.train):,} | avaliação ({args.eval_set}) {len(eval_df):,} | n_boot {args.n_boot} | GBS={include_gbs}")
    preds_raw, (cox, cox_names) = _train_all(
        pre, split, args.eval_set, config, include_gbs=include_gbs)

    # --- 1. Tabela comparativa ---
    preds = [ModelPrediction(name, r[0], r[1])
             for name, r in preds_raw.items()]
    table = comparison_table(
        preds, dur_e, ev_e, y_train, tau, n_boot=args.n_boot)
    print("\n=== Tabela comparativa (ordenada por C-Uno) ===")
    cols = ["model", "harrell", "harrell_lo", "harrell_hi", "uno", "uno_lo", "uno_hi", "antolini", "ibs",
            "brier_12", "brier_36", "brier_60", "brier_120"]
    print(table[cols].to_string(index=False,
          float_format=lambda x: f"{x:.4f}"))

    top = table.iloc[0]["model"]
    risk_top, surv_top, _ = preds_raw[top]
    print(f"\nmodelo de referência para calibração/era/sensibilidade: {top}")

    # --- 2. Calibração por decil (horizonte 5 anos) ---
    print(
        f"\n=== Calibração por decil de risco — S({CALIBRATION_HORIZON}m) ===")
    print(calibration_by_decile(surv_top, dur_e, ev_e, CALIBRATION_HORIZON).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    # --- 3. Métricas por era ---
    print("\n=== Métricas estratificadas por era ===")
    print(metrics_by_era(era_e, dur_e, ev_e, risk_top, surv_top, y_train,
          tau).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- 4. Validação temporal (Cox): 2010-2015 -> 2016-2017 ---
    print("\n=== Validação temporal (Cox): treinar 2010-2015, testar 2016-2017 ===")
    tv = temporal_validation_cox(df, ["2010-2015"], ["2016-2017"])
    print(
        f"  n_train={tv['n_train']:,} n_eval={tv['n_eval']:,} | Harrell temporal = {tv['harrell_temporal']:.4f}")
    print("  (comparar com o Harrell do split aleatório acima; se não desabar, era não carrega o resultado.)")

    # --- 5. Sensibilidade do tempo desconhecido (Cox) ---
    print("\n=== Sensibilidade: reincluir tempo desconhecido como óbito em t=0 ===")
    # A coorte de tempo desconhecido vem crua; limpa as features (alvo dummy só para o
    # clean_features montar o frame) e aplica o MESMO transformer ajustado no treino.
    unk = unknown.copy()
    unk["duration"], unk["event"] = 0.0, 1
    unk = clean_features(unk)
    x_unknown = pre.transformer.transform(
        unk[INPUT_COLUMNS]).astype(np.float32)
    fr_unknown = build_cox_frame(x_unknown, cox_names, np.zeros(
        len(x_unknown)), np.ones(len(x_unknown)))
    risk_unknown = cox_risk(cox, fr_unknown)
    risk_cox_eval = preds_raw["Cox"][0]
    sens = unknown_time_sensitivity(dur_e, ev_e, risk_cox_eval, risk_unknown)
    print(f"  n_desconhecido={sens['n_unknown']:,} | Harrell base={sens['base']:.4f} "
          f"com desconhecido={sens['with_unknown']:.4f} | delta={sens['delta']:+.4f}")

    print("\n=== Etapa 8 (dry-run no val): pipeline de avaliação completo executado. ===" if args.eval_set == "val"
          else "\n=== Etapa 8 FINAL concluída (teste aberto uma vez). ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
