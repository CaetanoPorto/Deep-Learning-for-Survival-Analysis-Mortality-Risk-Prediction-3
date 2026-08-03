# iniciar-claude.ps1
# Abre o Claude Code no repositorio de codigo com acesso ao vault Obsidian do TCC.
# Uso:  clique com o botao direito > "Executar com PowerShell"
#       ou no terminal:  .\iniciar-claude.ps1

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Ajuste estes dois caminhos se as pastas mudarem de lugar
# ---------------------------------------------------------------------------
$REPO  = "D:\User\Documentos\TCC_Analise_Sobrev\Claude_Code_TCC"
$VAULT = "D:\User\Documentos\Memoria Tcc Vault\TCC-Vault"

# ---------------------------------------------------------------------------
# Verificacoes
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== TCC - Analise de Sobrevivencia (SEER) ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
    Write-Host "[X] Claude Code nao encontrado no PATH." -ForegroundColor Red
    Write-Host "    Instale e reabra o PowerShell."
    Read-Host "Enter para sair"
    exit 1
}

if (-not (Test-Path $REPO)) {
    Write-Host "[X] Repositorio nao encontrado:" -ForegroundColor Red
    Write-Host "    $REPO"
    Write-Host "    Talvez seja a pasta 'Claude_Code_TCC - Copia'. Edite a variavel"
    Write-Host "    `$REPO no topo deste script."
    Read-Host "Enter para sair"
    exit 1
}

if (-not (Test-Path $VAULT)) {
    Write-Host "[X] Vault nao encontrado:" -ForegroundColor Red
    Write-Host "    $VAULT"
    Read-Host "Enter para sair"
    exit 1
}

$runbook = Join-Path $VAULT "09-Execucao\RUNBOOK - Pipeline Completo.md"
if (-not (Test-Path $runbook)) {
    Write-Host "[!] Aviso: o RUNBOOK nao foi encontrado dentro do vault." -ForegroundColor Yellow
    Write-Host "    O vault pode estar incompleto."
}

Write-Host "[ok] Claude Code : $((claude --version) 2>&1)" -ForegroundColor Green
Write-Host "[ok] Repositorio : $REPO"                      -ForegroundColor Green
Write-Host "[ok] Vault       : $VAULT"                     -ForegroundColor Green
Write-Host ""

# ---------------------------------------------------------------------------
# Configura e abre
# ---------------------------------------------------------------------------
# Faz o CLAUDE.md do vault tambem ser carregado (sem isso, --add-dir da acesso
# aos arquivos mas ignora as instrucoes de la).
$env:CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD = "1"

Set-Location $REPO

Write-Host "Lembretes:" -ForegroundColor Yellow
Write-Host "  1. Rode /context e confirme que aparecem DOIS CLAUDE.md."
Write-Host "  2. Uma sessao = uma etapa do RUNBOOK. Use /clear entre elas."
Write-Host "  3. Os prompts prontos estao em INICIAR.md."
Write-Host ""

claude --add-dir "$VAULT"
