# Full A3 protocol on a local Windows GPU, using an existing Python env.
# Layout expected (defaults match the team machine):
#   <Root>\a3_run\PR-CLIP           authors' code @242fba6, already patched (apply_patches.py)
#   <Root>\a3_run\PR-CLIP-Improved  this repository
#   <Root>\work\venv\Scripts\python.exe, <Root>\work\deps (PYTHONPATH with torch+CUDA)
#   <Root>\work\images\image_all, <Root>\work\RS5M_ViT-B-32_RET-2.pt, <Root>\work\openclip-cache
# Finished runs are skipped, so the script can be started again after an interruption.
param(
    [string]$Root = (Resolve-Path "$PSScriptRoot\..\..\..").Path,
    [int[]]$Seeds = @(1, 2, 3),
    [string[]]$TrainCfgs = @("B1", "R2", "R3"),
    [string]$Workers = "2"
)
$ErrorActionPreference = "Stop"
$Run = Join-Path $Root "a3_run"
$Repo = Join-Path $Run "PR-CLIP"
$Imp = Join-Path $Run "PR-CLIP-Improved"
$Work = Join-Path $Root "work"
$Py = Join-Path $Work "venv\Scripts\python.exe"
$Images = Join-Path $Work "images\image_all"
$Ckpt = Join-Path $Work "RS5M_ViT-B-32_RET-2.pt"
$Cache = Join-Path $Work "openclip-cache"
$Logs = Join-Path $Work "a3_logs"
New-Item -ItemType Directory -Force -Path $Logs, (Join-Path $Work "tmp") | Out-Null
$env:PYTHONPATH = Join-Path $Work "deps"
$env:HF_HOME = Join-Path $Work "hf-cache"
$env:TEMP = Join-Path $Work "tmp"; $env:TMP = $env:TEMP
$env:PYTHONIOENCODING = "utf-8"
Start-Transcript -Path (Join-Path $Run "run_log.txt") -Append | Out-Null

& $Py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
& $Py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA is not available in $Py" }

if (-not (Test-Path (Join-Path $Repo "dataset\val\ret2_val.csv"))) {
    & $Py (Join-Path $Imp "scripts\make_val_split.py") --repo $Repo --val-frac 0.1 --seed 0
    if ($LASTEXITCODE -ne 0) { throw "val split failed" }
}
Copy-Item (Join-Path $Repo "dataset\split_manifest.json") (Join-Path $Logs "split_manifest.json") -Force

function Done($pattern) {
    foreach ($d in (Get-ChildItem -Path $Logs -Directory -Filter "$pattern*" -ErrorAction SilentlyContinue)) {
        $f = Join-Path $d.FullName "metrics.jsonl"
        if ((Test-Path $f) -and ((Get-Item $f).Length -gt 0)) { return $true }
    }
    return $false
}
function TrainDone($tag) {
    foreach ($d in (Get-ChildItem -Path $Logs -Directory -Filter "$tag*" -ErrorAction SilentlyContinue)) {
        $f = Join-Path $d.FullName "metrics.jsonl"
        if ((Test-Path $f) -and (Select-String -Path $f -Pattern '"epoch": 1' -Quiet)) { return $true }
    }
    return $false
}
function Run([string[]]$a) {
    Write-Host ("`n>> " + (Get-Date -Format "HH:mm:ss") + " run_a3.py " + ($a -join " "))
    & $Py (Join-Path $Imp "scripts\run_a3.py") @a --repo $Repo --logs $Logs --images $Images --ckpt $Ckpt --cache-dir $Cache --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "run failed: $($a -join ' ')" }
}

# Phase 1: epoch 0 (no training), B1 vs RIP-fixed
foreach ($cfg in @("B1", "R2")) {
    foreach ($data in @("rsicd_test", "rsitmd_test")) {
        if (-not (Done "A3_${cfg}_ep0_${data}_s1")) { Run @("eval", "--cfg", $cfg, "--data", $data, "--seed", "1") }
    }
}

# Phase 2: one epoch per config and seed; val before/after training; test with epoch-1 checkpoint
foreach ($seed in $Seeds) {
    foreach ($cfg in $TrainCfgs) {
        $tag = "A3_${cfg}_train_s$seed"
        if (-not (TrainDone $tag)) {
            # remove an interrupted (or pre-AMP-FIX) attempt so its epoch-0 line is not reused
            Get-ChildItem -Path $Logs -Directory -Filter "$tag*" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
            Run @("train", "--cfg", $cfg, "--seed", "$seed")
        }
        $ck = Get-ChildItem -Path $Logs -Recurse -Filter "epoch_1.pt" -ErrorAction SilentlyContinue |
              Where-Object { $_.FullName -like "*\$tag*" } | Select-Object -Last 1
        foreach ($data in @("rsicd_test", "rsitmd_test")) {
            if (-not (Done "A3_${cfg}_ep1_${data}_s$seed")) {
                if (-not $ck) { throw "missing epoch_1.pt for $tag" }
                Run @("eval", "--cfg", $cfg, "--data", $data, "--seed", "$seed", "--resume", $ck.FullName)
            }
        }
        Get-ChildItem -Path $Logs -Recurse -Include "*.pt", "*.tmp" -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like "*\$tag*" } | Remove-Item -Force
        Write-Host ("DONE " + $cfg + " seed " + $seed + " at " + (Get-Date -Format "HH:mm:ss"))
    }
}

# Phase 3: tables
& $Py (Join-Path $Imp "scripts\aggregate_results.py") --logs $Logs --out $Imp --test rsicd_test
& $Py (Join-Path $Imp "scripts\aggregate_results.py") --logs $Logs --out (Join-Path $Logs "rsitmd_summary") --test rsitmd_test
Write-Host "ALL DONE"
Stop-Transcript | Out-Null
