# TCC — Análise de Sobrevivência de Câncer de Mama (SEER) com Cox, DeepSurv e DeepHit

Pipeline reprodutível e à prova de vazamento de dados para comparar um baseline Cox
proporcional clássico (lifelines) com dois modelos de Deep Learning para análise de
sobrevivência — DeepSurv e DeepHit (pycox/PyTorch) — na base SEER de câncer de mama
(17 Registries, 2000–2022, ~1,36M linhas).

## Dados

- `breast_cancer.dic`: dicionário SEER*Stat com o nome completo das 22 colunas exportadas.
- `TCC_ML_DL-20260707T020648Z-3-001/TCC_ML_DL/breast_cancer.csv`: a base bruta (~300MB).
- Alvo: `Survival months` (tempo) + `Vital status recode (study cutoff used)`
  (evento: Dead=1, Alive=0/censurado).

## Setup

### Local (Windows/PowerShell ou Git Bash)

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Todos os comandos abaixo usam `./.venv/Scripts/python.exe` (ou ative o venv:
`./.venv/Scripts/activate` no PowerShell, depois use `python` normalmente).

### Google Colab

```python
!pip install -r requirements.txt
```

Faça upload (ou monte o Google Drive com) `breast_cancer.csv`/`breast_cancer.dic` nos
mesmos caminhos relativos usados em `src/config.py` (`RAW_CSV_PATH`/`RAW_DIC_PATH`), ou
ajuste essas duas constantes para o caminho no Drive. O resto do código roda sem
alterações — não há nada específico de SO além dos caminhos.

## Rodando o pipeline

Todo script aceita `--sample-n N` (amostra aleatória de N linhas, default 50.000 — bom
para prototipar em segundos/poucos minutos) ou `--full` (dataset completo, ~1,36M linhas
— demora mais e usa mais memória; só rodar depois de validar em amostra).

```bash
# 1. Só o pipeline de dados/preprocessing, com relatório de sanidade anti-vazamento
python scripts/sanity_check_data.py

# 2. Baseline Cox
python scripts/run_baseline.py --sample-n 50000
python scripts/run_baseline.py --full

# 3. DeepSurv
python scripts/run_deepsurv.py --sample-n 50000

# 4. DeepHit
python scripts/run_deephit.py --sample-n 50000

# 5. Avaliação final: treina os três no MESMO split e compara
python scripts/run_evaluation.py --sample-n 50000
python scripts/run_evaluation.py --full
```

O script 5 é o que produz a tabela comparativa final (C-index de Harrell, C-index de
Antolini tempo-dependente, Brier score integrado/IBS) — os scripts 2-4 existem para
inspecionar cada modelo isoladamente (curva de loss, LR sugerido, etc.) sem re-treinar
os outros dois.

## Testes

```bash
./.venv/Scripts/python.exe -m pytest tests/ -v
```

Cobrem principalmente a regra anti-vazamento: limpeza de "falsos nulos"/sentinelas do
SEER, unificação dos pares de colunas antigas/novas, ausência de `duration`/`event` nas
features após o preprocessing, e duas regressões específicas (ver "Decisões e
armadilhas" abaixo).

## Estrutura

```
src/
  config.py            seeds, caminhos, esquema de colunas, hiperparâmetros default
  data/                 loading (CSV -> DataFrame), cleaning (nulos/sentinelas/combine_first), target (duration/event)
  preprocessing/        split treino/val/teste estratificado, ColumnTransformer (fit só no treino)
  models/                cox_baseline.py, deepsurv.py, deephit.py
  train/                 lr_finder.py, trainer.py (compartilhados por DeepSurv/DeepHit)
  evaluate/              metrics.py (C-index Harrell/Antolini, Brier/IBS), sanity_checks.py, report.py
scripts/                 pontos de entrada executáveis (um por etapa do pipeline)
tests/                   pytest
```

## Decisões metodológicas (para a banca)

- **Split antes de qualquer imputação/normalização.** `preprocessing/split.py` divide
  treino/val/teste (estratificado por `event`) ANTES de `preprocessing/pipeline.py`
  ajustar (`fit`) imputadores/scaler/one-hot — que só veem o treino. Val/teste só
  passam por `transform`. Essa ordem é o que evita o vazamento que gerava C-index=1.0
  no rascunho anterior.
- **Sexo.** A coorte foi restrita a `Sex == Female` (câncer de mama masculino é
  ~0,75% da base e biologicamente distinto o suficiente para um estudo à parte).
- **"Falsos nulos" e códigos sentinela do SEER** (`Blank(s)`, `Unknown`, `Recode not
  available`, códigos numéricos como 95-99 em linfonodos ou 999 em tamanho de tumor)
  viram `NaN` explicitamente em `data/cleaning.py` antes de qualquer outra
  transformação — sem isso eles seriam tratados como categorias/magnitudes reais.
- **Pares de colunas antigas/novas do SEER** (ER, PR, HER2, grau, tamanho do tumor)
  são unificados via `combine_first` (dando prioridade à variável mais nova), depois
  de harmonizar o vocabulário das duas (ex.: "ER positive" -> "Positive") — combinar
  sem harmonizar criaria categorias duplicadas.
- **One-hot com `drop="first"`.** Sem isso, os dummies de cada variável categórica
  somam sempre 1, criando colinearidade perfeita entre grupos — o que deixa a matriz
  de informação do Cox (lifelines) singular/instável na inversão.
- **C-index do DeepHit não usa `1 - S(t_max)`.** O último ponto da grade de tempo
  discretizada colapsa para perto de zero para quase todo mundo (artefato conhecido
  de modelos de tempo discreto), o que invertia o C-index (~0,21 em vez de ~0,79).
  `models/deephit.py::predict_risk` usa a soma da curva de sobrevivência inteira
  (aproximação do tempo médio restrito) como escore de risco — coberto por
  `tests/test_deephit.py`.
- **Shims de compatibilidade em `evaluate/metrics.py`.** O `pycox` 0.2.3 (não
  recebe atualização desde antes do pandas 2.0/scipy 1.14) chama `pd.Series.is_monotonic`
  e `scipy.integrate.simps`, ambos removidos nas versões instaladas. Os shims no topo
  do arquivo (`if not hasattr(...)`) resolvem isso sem tocar no código do pycox.
- **Relatório de sanidade.** `evaluate/sanity_checks.py::check_cindex_sanity` roda
  depois de cada treino e avisa se o C-index de teste for implausível (>0,95) ou se o
  gap treino-teste for grande (>0,05) — os limiares estão em `config.py`.
