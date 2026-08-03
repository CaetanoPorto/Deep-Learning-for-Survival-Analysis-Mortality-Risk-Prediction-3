# Como iniciar o Claude Code com o vault — passo a passo

> Objetivo: abrir o Claude Code no repositório de código **enxergando o vault Obsidian**,
> para que ele escreva o pipeline seguindo as regras do SEER que já estão documentadas lá.

---

## Parte 1 — Conferir o terreno (uma vez só)

Abra o **PowerShell** e rode:

```powershell
# 1. O Claude Code está instalado?
claude --version

# 2. Os dois caminhos existem?
Test-Path "D:\User\Documentos\TCC_Analise_Sobrev\Claude_Code_TCC"
Test-Path "D:\User\Documentos\Memoria Tcc Vault\TCC-Vault"
```

As duas últimas linhas precisam responder `True`. Se a primeira der `False`, ajuste o
caminho do repositório nos comandos abaixo (pode ser que a pasta real chame
`Claude_Code_TCC - Copia`).

> [!warning] Atenção ao "- Copia"
> Existem duas pastas parecidas: `Claude_Code_TCC` e `Claude_Code_TCC - Copia`.
> **Decida qual é a de trabalho e use sempre a mesma.** Os arquivos `CLAUDE.md` e
> `INICIAR.md` novos foram escritos na pasta que você conectou ao Cowork. Se a de
> trabalho for a outra, copie os dois arquivos para lá.

---

## Parte 2 — Abrir a sessão (toda vez)

```powershell
cd "D:\User\Documentos\TCC_Analise_Sobrev\Claude_Code_TCC"

$env:CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD = "1"

claude --add-dir "D:\User\Documentos\Memoria Tcc Vault\TCC-Vault"
```

O que cada linha faz:

| Linha | Por quê |
|---|---|
| `cd ...Claude_Code_TCC` | o Claude Code carrega o `CLAUDE.md` da pasta onde você o inicia |
| `$env:CLAUDE_CODE_..._CLAUDE_MD = "1"` | faz o `CLAUDE.md` **do vault** também ser carregado — sem isso, o `--add-dir` dá acesso aos arquivos mas ignora as instruções |
| `--add-dir "...TCC-Vault"` | dá ao Claude Code permissão de ler o vault inteiro |

Na primeira vez ele pode pedir confirmação para acessar a pasta extra. Aceite.

### Confirme que carregou

Dentro da sessão, digite:

```
/context
```

Em **Memory files** devem aparecer **dois** `CLAUDE.md`: o do repositório e o do vault.
Se só aparecer um, a variável de ambiente não pegou — feche e refaça a Parte 2.

---

## Parte 3 — Os prompts, em ordem

Não peça "escreva o projeto todo". Vá por sessões, cada uma com entrega verificável.
Copie e cole um bloco por vez.

---

### 🔍 Sessão 1 — Auditoria (não escreve código)

```
Leia, nesta ordem, do vault que está em --add-dir:

1. 09-Execucao\RUNBOOK - Pipeline Completo.md
2. 09-Execucao\Contrato de Dados.md
3. 02-Dados\Regras de Codificação SEER.md
4. 02-Dados\Blanks e Eras de Diagnóstico.md
5. 02-Dados\Perfil Empírico da Base.md
6. As 4 notas de 06-Armadilhas\

Depois leia todo o src/ e tests/ deste repositório.

Sua tarefa nesta sessão é SÓ auditoria. NÃO altere nenhum arquivo.

Produza um relatório em markdown com:
- Toda divergência entre o que src/ faz e o que o Contrato de Dados exige
- Para cada divergência: arquivo, linha, o que está errado, o que deveria ser,
  e o impacto (quantos registros afetados, usando os números do Perfil Empírico)
- Classificação por severidade: bloqueante / importante / cosmético
- Ordem de correção sugerida

Salve como AUDITORIA.md na raiz do repositório.
```

**Antes de seguir:** leia o `AUDITORIA.md`. Se ele inventou divergência ou citou número
que não está no Perfil Empírico, corrija o rumo agora.

---

### 🔧 Sessão 2 — Camada de dados

```
Com base no AUDITORIA.md e no Contrato de Dados, corrija a camada de dados:

1. src/config.py — endpoint all-cause, faixas de sentinela corretas, remover
   "Blank(s)" do FAKE_NULL_TOKENS global
2. Nova função de reconstrução de era_diagnostico, aplicada ANTES de qualquer limpeza
3. src/data/cleaning.py — decodificar sentinelas conforme ADR-005:
   node_count, node_status, nodes_examined_n, tumor_size_mm com ponto médio das
   faixas 991-997 e teto de 200 mm
4. Testes em tests/ para as 7 invariantes do Contrato de Dados

Critério de aceite (etapas 1 e 2 do RUNBOOK): rode os testes e me mostre que
N_TOTAL, N_FEMININO, N_COORTE_ANALITICA, ERAS e MAX_FOLLOWUP batem exatamente.

Se algum não bater, PARE e me explique — não ajuste o teste para passar.
```

---

### ✂️ Sessão 3 — Split e pré-processamento

```
Etapas 4 e 5 do RUNBOOK.

1. src/preprocessing/split.py — estratificar por event E era_diagnostico
2. src/preprocessing/pipeline.py — revisar o ColumnTransformer segundo o
   Contrato de Dados
3. Teste que detecta o vazamento da era: treinar um classificador simples
   X -> era_diagnostico e falhar se a acurácia for muito acima do baseline
   (ver 06-Armadilhas\Armadilha - Era de diagnóstico vaza pelo padrão de Blanks.md)

Critério de aceite: assert_no_target_leakage passa, diferença de taxa de censura
entre folds < 3%, e nenhum fit ocorre antes do split.
```

---

### 📊 Sessão 4 — Baselines

```
Etapa 6 do RUNBOOK, na ordem exata: Cox → teste de riscos proporcionais →
Cox+splines → Random Survival Forest → Gradient Boosting de sobrevivência.

RSF e GBS ainda não existem no repositório — crie src/models/rsf.py e
src/models/gbs.py com a mesma superfície dos outros modelos
(build_* / predict_risk / predict_survival_function).

Rode em --sample-n 50000 primeiro.

Critério de aceite por modelo: 0,55 < C-index de teste < 0,95 e gap treino-teste
< 0,05. Fora disso, PARE e consulte as notas de armadilha.
```

---

### 🧠 Sessão 5 — Modelos profundos

```
Etapa 7 do RUNBOOK: LR finder → DeepSurv → DeepHit.

Mínimo 3 sementes por modelo, registrando média e desvio.
Discretização do tempo por quantis, 20 bins, com bin que comporte t=0.

Critério de aceite: mesmos limiares da etapa 6, mais curva de perda de validação
que decresce e estabiliza.
```

---

### 📈 Sessão 6 — Avaliação final

```
Etapa 8 do RUNBOOK, seguindo 09-Execucao\Protocolo de Validação.md.

O conjunto de teste é aberto UMA vez. Produza:
1. Tabela modelo x {C-Harrell, C-Uno, C-Antolini, IBS, Brier@12/36/60/120} com IC bootstrap
2. Curvas de calibração por decil de risco
3. Métricas estratificadas por era
4. Validação temporal: treinar 2010-2015, testar 2016-2017
5. Análise de sensibilidade dos 6.052 tempos desconhecidos

Depois crie uma nota por rodada em 04-Experimentos\ do vault, usando
_templates\Template - Experimento.md.
```

---

## Parte 4 — Hábitos que fazem diferença

| Hábito | Por quê |
|---|---|
| `/context` no início | confirma que os dois `CLAUDE.md` carregaram |
| Uma sessão = uma etapa do RUNBOOK | contexto limpo, entrega verificável |
| `/clear` entre sessões | evita arrastar contexto velho e confundir o modelo |
| Leia o diff antes de aceitar | o Claude Code erra; o TCC é seu |
| Critério de aceite falhou → não ajuste o teste | ajustar o teste para passar é como o C-index 1.0 nasceu |
| Decisão nova → peça um ADR em `05-Decisoes\` | é o que alimenta a Metodologia depois |

## Parte 5 — Se algo der errado

| Sintoma | Causa provável | Solução |
|---|---|---|
| `/context` mostra só um `CLAUDE.md` | variável de ambiente não pegou | refazer a Parte 2 na mesma janela do PowerShell |
| "não encontro o vault" | caminho com espaço sem aspas | use aspas duplas em todo o caminho |
| Ele escreve código sem ler o vault | prompt genérico demais | cite o arquivo exato: "leia `09-Execucao\Contrato de Dados.md`" |
| Números não batem com o Perfil Empírico | bug no código, ou CSV diferente | **pare** — não siga com números divergentes |
| Ele "conserta" o pycox e quebra tudo | os monkeypatches de `metrics.py` | mande ler os comentários antes de mexer |
