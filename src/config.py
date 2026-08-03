"""Configuração central: caminhos, schema literal das 22 colunas do SEER, regra de
reconstrução da era de diagnóstico e os números medidos do Perfil Empírico da Base.

Mantido como fonte única de constantes para que nenhum módulo use um nome de coluna,
seed ou limiar divergente. Os nomes de coluna são **literais** (copiados do cabeçalho
do CSV / Dicionário de Variáveis) — não normalizar acentos, parênteses ou espaços.

Este arquivo cresce etapa a etapa junto com o RUNBOOK. Nesta primeira leva estão as
constantes das Etapas 0–1 (perfil empírico, coorte, reconstrução de era). Sentinelas,
pares redundantes, agrupamentos e hiperparâmetros entram nas suas respectivas etapas.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------

RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Fixa o seed em todas as fontes de aleatoriedade. Chamar no início de cada
    script (não em import time) para que cada execução registre o seed no próprio log.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# O CSV do SEER fica FORA do repositório Git (guardrail: nunca versionar). Localmente
# está no diretório exportado do Drive; no Colab (ou outra máquina), defina a variável de
# ambiente SEER_CSV_PATH apontando para o CSV — não precisa editar este arquivo.
_DEFAULT_CSV = PROJECT_ROOT / "TCC_ML_DL-20260707T020648Z-3-001" / "TCC_ML_DL" / "breast_cancer.csv"
RAW_CSV_PATH = Path(os.environ.get("SEER_CSV_PATH", _DEFAULT_CSV))
RAW_DIC_PATH = PROJECT_ROOT / "breast_cancer.dic"

# Tudo lido como string, sem inferência de nulo do pandas: a semântica de "vazio" é
# decidida pelo código (há três tipos diferentes de vazio no SEER — ver Regras de
# Codificação SEER), não pelo parser.
READ_CSV_KWARGS: dict = dict(dtype=str, keep_default_na=False, na_values=[])

# ---------------------------------------------------------------------------
# Schema bruto — as 22 colunas na ordem do CSV (Dicionário de Variáveis)
# ---------------------------------------------------------------------------

# Desfecho (labels — nunca entram na matriz de features)
COL_SURVIVAL_MONTHS = "Survival months"
COL_VITAL_STATUS = "Vital status recode (study cutoff used)"

# Filtro de coorte
COL_SEX = "Sex"

# Demográficas
COL_AGE = "Age recode with <1 year olds and 90+"
COL_RACE = "Race recode (W, B, AI, API)"

# Grau (par redundante por era)
COL_GRADE_THRU_2017 = "Grade Recode (thru 2017)"
COL_GRADE_2018 = "Derived Summary Grade 2018 (2018+)"

# Estadiamento e subtipo
COL_STAGE = "Combined Summary Stage with Expanded Regional Codes (2004+)"
COL_BREAST_SUBTYPE = "Breast Subtype (2010+)"

# Receptores (três pares redundantes)
COL_ER_OLD = "ER Status Recode Breast Cancer (1990+)"
COL_ER_NEW = "Estrogen Receptor Summary (2018+)"
COL_PR_OLD = "PR Status Recode Breast Cancer (1990+)"
COL_PR_NEW = "Progesterone Receptor Summary (2018+)"
COL_HER2_OLD = "Derived HER2 Recode (2010+)"
COL_HER2_NEW = "HER2 Overall Summary Recode (2018+)"

# Tamanho do tumor (par redundante)
COL_TUMOR_SIZE_CS = "CS tumor size (2004-2015)"
COL_TUMOR_SIZE_SUMMARY = "Tumor Size Summary (2016+)"

# Linfonodos
COL_NODES_POSITIVE = "Regional nodes positive (1988+)"
COL_NODES_EXAMINED = "Regional nodes examined (1988+)"

# Tratamento
COL_SURGERY = "RX Summ--Surg Prim Site (1998+)"
COL_RADIATION = "Radiation recode"
COL_CHEMOTHERAPY = "Chemotherapy recode (yes, no/unk)"

# Ordem exata do cabeçalho do CSV — validada em scripts/profile_dataset.py (Etapa 0).
RAW_COLUMNS: list[str] = [
    COL_SURVIVAL_MONTHS,
    COL_VITAL_STATUS,
    COL_AGE,
    COL_SEX,
    COL_RACE,
    COL_GRADE_THRU_2017,
    COL_GRADE_2018,
    COL_STAGE,
    COL_BREAST_SUBTYPE,
    COL_ER_OLD,
    COL_PR_OLD,
    COL_HER2_OLD,
    COL_ER_NEW,
    COL_PR_NEW,
    COL_HER2_NEW,
    COL_TUMOR_SIZE_CS,
    COL_TUMOR_SIZE_SUMMARY,
    COL_NODES_POSITIVE,
    COL_NODES_EXAMINED,
    COL_SURGERY,
    COL_RADIATION,
    COL_CHEMOTHERAPY,
]

# ---------------------------------------------------------------------------
# Desfecho e coorte
# ---------------------------------------------------------------------------

# Endpoint = sobrevivência global (all-cause). Dead=1 (evento), Alive=0 (censurado no
# corte 31/12/2022). NÃO é cause-specific — o export não tem causa do óbito (ADR-001).
EVENT_MAP: dict[str, int] = {"Dead": 1, "Alive": 0}

# Token textual de tempo desconhecido em `Survival months` (6.052 casos, 100% óbitos).
SURVIVAL_UNKNOWN_TOKEN = "Unknown"

SEX_KEEP_VALUE = "Female"

# ---------------------------------------------------------------------------
# Reconstrução da era de diagnóstico (Blanks e Eras de Diagnóstico / ADR-003)
# ---------------------------------------------------------------------------
# O export não traz `Year of diagnosis`; o ano está codificado no padrão de `Blank(s)`
# das variáveis específicas de era. A era é confundidor estrutural: entra como estrato
# do split e como restrição de coorte, NUNCA como feature preditiva.

BLANK_TOKEN = "Blank(s)"
BREAST_SUBTYPE_NA_TOKEN = "Recode not available"  # "blank" estrutural do subtipo pré-2010

ERA_2000_2003 = "2000-2003"
ERA_2004_2009 = "2004-2009"
ERA_2010_2015 = "2010-2015"
ERA_2016_2017 = "2016-2017"
ERA_2018_2022 = "2018-2022"

ERA_LABELS: list[str] = [
    ERA_2000_2003,
    ERA_2004_2009,
    ERA_2010_2015,
    ERA_2016_2017,
    ERA_2018_2022,
]

# Follow-up máximo (meses) observável por era até o corte de 31/12/2022. É a validação
# decisiva da reconstrução: cinco coincidências exatas com o máximo de `Survival months`.
ERA_MAX_FOLLOWUP: dict[str, int] = {
    ERA_2000_2003: 275,
    ERA_2004_2009: 227,
    ERA_2010_2015: 155,
    ERA_2016_2017: 83,
    ERA_2018_2022: 59,
}

# ---------------------------------------------------------------------------
# Split treino/validação/teste
# ---------------------------------------------------------------------------

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
CENSORING_RATE_TOLERANCE = 0.03  # diferença máx. de taxa de censura entre folds
ERA_BALANCE_TOLERANCE = 0.01  # diferença máx. de proporção de uma era entre folds

# ---------------------------------------------------------------------------
# Limiares de sanidade anti-vazamento (Protocolo de Validação)
# ---------------------------------------------------------------------------

SANITY_CINDEX_MAX = 0.95
SANITY_CINDEX_MIN = 0.55
SANITY_TRAIN_TEST_GAP_MAX = 0.05

# ===========================================================================
# Perfil Empírico da Base — números MEDIDOS (não estimados), varridos do CSV em
# 22/07/2026. Fonte: nota "Perfil Empírico da Base" + invariantes do "Contrato de
# Dados". Se o CSV for re-exportado e estes números não baterem, a Etapa 0 falha e o
# pipeline PARA (não prossegue com números divergentes — RUNBOOK Etapa 0).
# ===========================================================================

EXPECTED_N_ROWS = 1_365_329

EXPECTED_VITAL_STATUS_COUNTS: dict[str, int] = {"Alive": 920_439, "Dead": 444_890}

EXPECTED_SURVIVAL_UNKNOWN = 6_052  # `Survival months == "Unknown"`, todos `Dead`
EXPECTED_SURVIVAL_ZERO = 16_918  # `Survival months == "0000"` (8.482 Dead / 8.436 Alive)

EXPECTED_SEX_COUNTS: dict[str, int] = {"Female": 1_355_045, "Male": 10_284}

EXPECTED_ERA_COUNTS: dict[str, int] = {
    ERA_2000_2003: 205_049,
    ERA_2004_2009: 322_229,
    ERA_2010_2015: 361_127,
    ERA_2016_2017: 130_721,
    ERA_2018_2022: 346_203,
}

# Invariante dos linfonodos: `nodes_positive == "98"` ⟺ `nodes_examined == "00"`,
# em exatamente 198.635 registros, sem uma única exceção.
EXPECTED_NODE_INVARIANT_COUNT = 198_635

# Complementaridade perfeita do grau: as duas colunas nunca estão ambas preenchidas
# nem ambas vazias; a soma dos `Blank(s)` de cada uma é N.
EXPECTED_GRADE_BLANK_COUNTS: dict[str, int] = {
    COL_GRADE_THRU_2017: 346_203,  # Blank(s) exatamente na era 2018–2022
    COL_GRADE_2018: 1_019_126,  # Blank(s) em todas as eras anteriores a 2018
}

# --- Fluxo de coorte (Critérios de Inclusão e Exclusão — contagens medidas) ---
# Base bruta 1.365.329 − 10.284 (homens) = 1.355.045 (coorte feminina)
#            1.355.045 − 5.988 (tempo desconhecido, feminino) = 1.349.057 (analítica)
# Dos 6.052 tempos desconhecidos totais, 64 são homens (já saem no filtro de sexo).
# NB: o pipeline reproduz 1.349.057 (medido); a ADR-004 traz "≈1.349.015" e será
# atualizada pelo Guilherme para casar com este número.
EXPECTED_UNKNOWN_TIME_TOTAL = 6_052
EXPECTED_UNKNOWN_TIME_MALE = 64
EXPECTED_UNKNOWN_TIME_FEMALE = 5_988  # removidos da coorte feminina, 100% óbitos
EXPECTED_ANALYTIC_COHORT = 1_349_057

# ===========================================================================
# Etapa 2 — limpeza SEER (schema de saída em Contrato de Dados / ADR-005)
# ===========================================================================

# --- Pares redundantes antigo/novo (combine_first, prioridade ao mais novo) ---
# O vocabulário é harmonizado ANTES de combinar (senão "Positive" e "ER positive" viram
# níveis distintos). O `Blank(s)` estrutural é tratado como ausente no combine — a era
# já foi reconstruída, então isso é legítimo (Blanks e Eras de Diagnóstico).


@dataclass(frozen=True)
class RedundantPair:
    old_col: str
    new_col: str
    output_col: str
    old_value_map: dict[str, str] = field(default_factory=dict)
    new_value_map: dict[str, str] = field(default_factory=dict)


GRADE_VALUE_MAP = {
    "Well differentiated; Grade I": "1",
    "Moderately differentiated; Grade II": "2",
    "Poorly differentiated; Grade III": "3",
    "Undifferentiated; anaplastic; Grade IV": "4",
}

REDUNDANT_PAIRS: list[RedundantPair] = [
    RedundantPair(COL_ER_OLD, COL_ER_NEW, "er_status",
                  new_value_map={"ER positive": "Positive", "ER negative": "Negative"}),
    RedundantPair(COL_PR_OLD, COL_PR_NEW, "pr_status",
                  new_value_map={"PR positive": "Positive", "PR negative": "Negative"}),
    RedundantPair(COL_HER2_OLD, COL_HER2_NEW, "her2_status",
                  new_value_map={"HER2 positive": "Positive", "HER2 negative; equivocal": "Negative"}),
    RedundantPair(COL_GRADE_THRU_2017, COL_GRADE_2018, "tumor_grade",
                  old_value_map=GRADE_VALUE_MAP),
    RedundantPair(COL_TUMOR_SIZE_CS, COL_TUMOR_SIZE_SUMMARY, "tumor_size_mm"),
]

# --- Tokens de nulo REAIS (informação não obtida) -> NaN nas categóricas. ---
# NÃO inclui Blank(s) (estrutural, já resolvido pelo combine) nem os estados clínicos
# de radiação (None/Unknown, Recommended..., Refused), que viram grupo próprio.
REAL_NULL_TOKENS: set[str] = {
    "Unknown",
    "Recode not available",
    "Borderline/Unknown",
    "Not documented; Cannot be determined; Not assessed or unknown if assessed",
    "Not documented; Indeterminate; Not assessed or unknown if assessed",
    "Test ordered, results not in chart",
}
# Grau 2018+: "9" = desconhecido; letras = sistemas de graduação alternativos sem
# equivalência 1:1 com Nottingham -> tratados como faltante (ADR-005; NÃO inventar mapa).
GRADE_NULL_TOKENS: set[str] = {"9", "A", "B", "C", "D", "H", "L", "M"}

# --- Linfonodos (ADR-005): três variáveis derivadas ---
NODE_COUNT_MAX = 90  # 00-90 são contagens; 95/96/97/98/99 são códigos
NODE_STATUS_NEGATIVE = "negativo"
NODE_STATUS_POSITIVE = "positivo"
NODE_STATUS_NOT_ASSESSED = "nao_avaliado"
NODE_STATUS_UNKNOWN = "desconhecido"
NODE_STATUS_LEVELS = [
    NODE_STATUS_NEGATIVE, NODE_STATUS_POSITIVE, NODE_STATUS_NOT_ASSESSED, NODE_STATUS_UNKNOWN,
]

# --- Tamanho do tumor (ADR-005): recuperar faixas do esquema CS + teto de plausibilidade ---
TUMOR_SIZE_CEILING_MM = 200  # sem `Primary Site` no export, teto defensável p/ mama
# Ponto médio (mm) das faixas 991-997 do esquema CS. 992->15 e 993->25 são explícitos no
# vault; 991/994/995 seguem o mesmo padrão de ponto médio por faixa de 1 cm.
# 996 (5-10 cm) e 997 (>10 cm) ficam de fora até confirmar o valor oficial do recode do
# SEER — NÃO inventar; viram NaN (384 registros, 0,03%). Ver checkpoint da Etapa 2.
CS_RANGE_MIDPOINTS_MM: dict[int, int] = {991: 5, 992: 15, 993: 25, 994: 35, 995: 45}
TUMOR_SIZE_MICROSCOPIC = 990  # foco microscópico -> NaN (não é medida)
TUMOR_SIZE_DIFFUSE = 998  # doença difusa/inflamatória -> NaN (+ flag)
TUMOR_SIZE_UNKNOWN = 999

# --- Idade -> ponto médio numérico (Plano de Modelagem: idade é ~contínua) ---
AGE_MIDPOINTS: dict[str, float] = {
    "00 years": 0,
    "01-04 years": 2.5,
    "05-09 years": 7,
    "10-14 years": 12,
    "15-19 years": 17,
    "20-24 years": 22,
    "25-29 years": 27,
    "30-34 years": 32,
    "35-39 years": 37,
    "40-44 years": 42,
    "45-49 years": 47,
    "50-54 years": 52,
    "55-59 years": 57,
    "60-64 years": 62,
    "65-69 years": 67,
    "70-74 years": 72,
    "75-79 years": 77,
    "80-84 years": 82,
    "85-89 years": 87,
    "90+ years": 92,
}

# --- Cirurgia: 48 códigos -> 4 famílias (Regras SEER especifica 20-24 e 40-59) ---
SURGERY_NONE = "nenhuma"
SURGERY_CONSERVATIVE = "conservadora"  # códigos 20-24 (mastectomia parcial/lumpectomia)
SURGERY_MASTECTOMY = "mastectomia"  # códigos 40-59 (simples/radical modificada/radical)
SURGERY_OTHER_UNKNOWN = "outra_desconhecida"  # 00 nenhuma cai à parte; resto (10-19,30-39,60-90,99)
SURGERY_GROUP_LEVELS = [
    SURGERY_NONE, SURGERY_CONSERVATIVE, SURGERY_MASTECTOMY, SURGERY_OTHER_UNKNOWN,
]

# --- Radiação: 8 categorias -> 4 níveis ---
RADIATION_NONE_UNKNOWN = "nenhuma_desconhecida"
RADIATION_BEAM = "feixe"
RADIATION_IMPLANT = "implante_isotopo"
RADIATION_REFUSED = "recusou"
RADIATION_GROUP_LEVELS = [
    RADIATION_NONE_UNKNOWN, RADIATION_BEAM, RADIATION_IMPLANT, RADIATION_REFUSED,
]
# "Radiation NOS" e "Combination" = radiação administrada, modalidade não separável nos
# 4 grupos -> feixe (radiação externa é a modalidade dominante). Escolha registrada no
# checkpoint da Etapa 2 para virar ADR.
# Textos exatos como aparecem no CSV (verificados na base — inclui o espaço duplo em
# "Radiation, NOS  method...").
RADIATION_GROUP_MAP: dict[str, str] = {
    "None/Unknown": RADIATION_NONE_UNKNOWN,
    "Recommended, unknown if administered": RADIATION_NONE_UNKNOWN,
    "Beam radiation": RADIATION_BEAM,
    "Radiation, NOS  method or source not specified": RADIATION_BEAM,
    "Combination of beam with implants or isotopes": RADIATION_BEAM,
    "Radioactive implants (includes brachytherapy) (1988+)": RADIATION_IMPLANT,
    "Radioisotopes (1988+)": RADIATION_IMPLANT,
    "Refused (1988+)": RADIATION_REFUSED,
}

# --- Schema final de features (saída da Etapa 2) ---
NUMERIC_FEATURES: list[str] = ["node_count", "nodes_examined_n", "tumor_size_mm", "age_mid"]
CATEGORICAL_FEATURES: list[str] = [
    "node_status", "race", "stage", "tumor_grade",
    "er_status", "pr_status", "her2_status",
    "surgery_group", "radiation_group", "chemotherapy",
]
# Ordinais (a ordem é usada na codificação da Etapa 5).
STAGE_ORDER = ["in_situ", "localized", "regional", "distant"]
GRADE_ORDER = ["1", "2", "3", "4"]

# --- Grupos de pré-processamento da Etapa 5 (fit SÓ no treino) ---
# Ordinais: codificação ordinal (preserva a ordem) + imputação pela mediana + escala.
ORDINAL_ENCODE: dict[str, list[str]] = {"stage": STAGE_ORDER, "tumor_grade": GRADE_ORDER}

# Categóricas nominais -> categoria "desconhecido" explícita (informativa) + one-hot.
# Inclui her2_status (ADR-009): apesar do missing dele ser estrutural de era pré-2010,
# tratá-lo como er/pr/race preserva o poder preditivo do HER2 na coorte principal
# (2010-2022), onde o missing é real e era-neutro — em vez de fabricar HER2 pela moda.
# Na coorte completa (exploração) "her2=desconhecido" indica pré-2010: confundidor
# documentado (ADR-003), tratado por restrição de coorte, não por imputação.
NOMINAL_DESCONHECIDO: list[str] = [
    "node_status", "race", "er_status", "pr_status", "her2_status",
    "surgery_group", "radiation_group", "chemotherapy",
]
DESCONHECIDO_LEVEL = "desconhecido"

# Numéricas/ordinais com missing estrutural de era (`tumor_size_mm`/`stage` pré-2004) vão
# pela mediana SEM indicador (era-safe). Preservar o "missing informativo" delas via
# indicador é decisão separada — ver escopo da ADR-009.

# breast_subtype é GUARDADO mas NÃO entra na matriz (decisão: usar ER/PR/HER2 separados;
# subtipo é função determinística deles — colinearidade). Fica no frame para referência.
KEPT_NOT_MODELED: list[str] = ["breast_subtype"]

# ===========================================================================
# Etapa 3 — restrição de era (coorte de análise, ADR-003)
# ===========================================================================
# A era é confundidor estrutural. A análise PRINCIPAL restringe a era; a exploração e as
# análises de sensibilidade usam a base completa. Default None = completa (exploração).
# O modelo FINAL só roda após o Guilherme/orientador confirmarem 2010-2022 (decisão em
# aberto — não hardcodar aqui até a confirmação).
ERA_RESTRICTION: list[str] | None = None

# Recomendação a confirmar (subtipo molecular disponível + prática clínica moderna).
# NB: ADR-003 cita "≈838.051" — essa é a contagem da BASE BRUTA de 2010-2022
# (361.127+130.721+346.203). Na coorte analítica (feminina, tempo conhecido) são
# 828.121 (356.865+129.245+342.011, medidos na Etapa 1). O pipeline reporta o real.
ERA_MAIN_COHORT_RECOMMENDED: list[str] = [ERA_2010_2015, ERA_2016_2017, ERA_2018_2022]

# ===========================================================================
# Etapa 6 — baselines (Plano de Modelagem)
# ===========================================================================


@dataclass
class CoxConfig:
    penalizer: float = 0.1  # ridge L2 — estabiliza ~100 colunas one-hot no lifelines


@dataclass
class SplineConfig:
    n_knots: int = 4
    degree: int = 3  # splines cúbicas nas contínuas (num__*) para captar não-linearidade


@dataclass
class RSFConfig:
    n_estimators: int = 100
    min_samples_leaf: int = 20  # folhas maiores = mais rápido e menos overfit
    max_features: str = "sqrt"


@dataclass
class GBSConfig:
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    subsample: float = 0.8


# C-index de Uno (IPCW): trunca em tau = percentil dos tempos de treino, para não
# instabilizar a ponderação pela censura na cauda (67% de censura — Protocolo de Validação).
UNO_TAU_PERCENTILE = 95

# Faixa de sanidade do C-index de teste (RUNBOOK Etapa 6). Fora disso: PARAR.
# (SANITY_CINDEX_MIN/MAX e SANITY_TRAIN_TEST_GAP_MAX já definidos acima.)

# Amostra fixa comum para a tabela comparativa final (ADR-010): dimensionada pelo teto de
# viabilidade do GBS. Todos os modelos comparados nela, com semente fixa.
COMPARISON_SAMPLE_N = 100_000

# ===========================================================================
# Etapa 7 — modelos profundos (DeepSurv, DeepHit — pycox)
# ===========================================================================


@dataclass
class MLPConfig:
    num_layers: int = 2
    hidden_units: int = 32
    dropout: float = 0.2
    batch_norm: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 100
    patience: int = 10  # early stopping por perda de validação
    learning_rate: float = 1e-3  # default; o LR finder sugere o real


@dataclass
class DeepHitConfig:
    num_time_bins: int = 20  # discretização do tempo por quantis (inclui bin p/ t=0)
    scheme: str = "quantiles"  # bins equidistantes ficariam vazios na cauda até 275 meses
    alpha: float = 0.2  # peso verossimilhança vs. ranking loss (Lee et al. 2018)
    sigma: float = 0.1  # temperatura da ranking loss

# Nº mínimo de sementes por modelo profundo (Plano de Modelagem / Protocolo de Validação).
MIN_SEEDS = 3

# ===========================================================================
# Etapa 8 — avaliação final (Protocolo de Validação)
# ===========================================================================
EVAL_HORIZONS = [12, 36, 60, 120]  # meses = 1, 3, 5, 10 anos (Brier)
CALIBRATION_HORIZON = 60  # curva de calibração por decil em 5 anos
N_BOOTSTRAP = 200  # reamostragens para IC (>= 200)
