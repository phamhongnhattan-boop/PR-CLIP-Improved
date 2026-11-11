# Run the full A3 protocol on a local Windows GPU (tested layout: RTX 4060 Laptop, 8 GB).
# Same steps as notebooks/A3_MultiSeed_Colab.ipynb. Finished runs are skipped.
#
# Usage (PowerShell, inside the Python env that has torch+CUDA):
#   .\scripts\run_a3_local.ps1 -PrClip C:\path\PR-CLIP -Images C:\path\images\image_all -Logs C:\path\a3_logs
param(
    [Parameter(Mandatory = $true)][string]$PrClip,   # clone of ADMIS-TONGJI/PR-CLIP (commit 242fba6)
    [Parameter(Mandatory = $true)][string]$Images,   # folder with all RSICD/RSITMD images (flat)
    [string]$Logs = "$PSScriptRoot\..\a3_logs",
    [int[]]$Seeds = @(1, 2, 3),
    [string[]]$TrainCfgs = @("B1", "R2", "R3")
)
$ErrorActionPreference = "Stop"
$Improved = (Resolve-Path "$PSScriptRoot\..").Path
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$Logs = (Resolve-Path $Logs).Path

# 1. patch authors' code, link images, checkpoint, validation split
python "$Improved\prclip_patches\apply_patches.py" $PrClip
if (-not (Test-Path "$PrClip\images")) { New-Item -ItemType Junction -Path "$PrClip\images" -Target $Images | Out-Null }
$ckpt = "$PrClip\cache\weights\RET-2.pt"
if (-not (Test-Path $ckpt)) {
    New-Item -ItemType Directory -Force -Path "$PrClip\cache\weights" | Out-Null
    Invoke-WebRequest "https://huggingface.co/Zilun/GeoRSCLIP/resolve/main/ckpt/RS5M_ViT-B-32_RET-2.pt" -OutFile $ckpt
}
python "$Improved\scripts\make_val_split.py" --repo $PrClip --val-frac 0.1 --seed 0
Copy-Item "$PrClip\dataset\split_manifest.json" "$Logs\split_manifest.json" -Force
python -m pytest -q "$Improved\tests"

function Done($pattern) {
    $f = Get-ChildItem -Path $Logs -Directory -Filter "$pattern*" -ErrorAction SilentlyContinue |
         ForEach-Object { Join-Path $_.FullName "metrics.jsonl" } | Where-Object { (Test-Path $_) -and ((Get-Item $_).Length -gt 0) }
    return [bool]$f
}
function Run([string[]]$a) {
    Write-Host ">> run_a3.py $($a -join ' ')"
    python "$Improved\scripts\run_a3.py" @a --repo $PrClip --logs $Logs
    if ($LASTEXITCODE -ne 0) { throw "run failed: $($a -join ' ')" }
}

# 2. epoch 0, no training
foreach ($cfg in @("B1", "R2")) {
    foreach ($data in @("rsicd_test", "rsitmd_test", "val")) {
        if (-not (Done "A3_${cfg}_ep0_${data}_s1")) { Run @("eval", "--cfg", $cfg, "--data", $data, "--seed", "1") }
    }
}

# 3. one epoch x seeds, test with epoch-1 checkpoint, then delete checkpoint
foreach ($cfg in $TrainCfgs) {
    foreach ($seed in $Seeds) {
        $tag = "A3_${cfg}_train_s$seed"
        if (-not (Done $tag)) { Run @("train", "--cfg", $cfg, "--seed", "$seed") }
        $ck = Get-ChildItem -Path $Logs -Recurse -Filter "epoch_1.pt" | Where-Object { $_.FullName -like "*$tag*" } | Select-Object -Last 1
        foreach ($data in @("rsicd_test", "rsitmd_test")) {
            if (-not (Done "A3_${cfg}_ep1_${data}_s$seed")) {
                if (-not $ck) { throw "missing checkpoint for $tag" }
                Run @("eval", "--cfg", $cfg, "--data", $data, "--seed", "$seed", "--resume", $ck.FullName)
            }
        }
        Get-ChildItem -Path $Logs -Recurse -Filter "*.pt" | Where-Object { $_.FullName -like "*$tag*" } | Remove-Item -Force
    }
}

# 4. tables + raw logs into the repo
python "$Improved\scripts\aggregate_results.py" --logs $Logs --out $Improved --test rsicd_test
python "$Improved\scripts\aggregate_results.py" --logs $Logs --out "$Logs\rsitmd_summary" --test rsitmd_test
Get-ChildItem -Path $Logs -Directory -Filter "A3_*" | ForEach-Object {
    $dst = Join-Path $Improved "results\a3\raw_logs\$($_.Name)"
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    foreach ($f in @("out.log", "metrics.jsonl", "params.yml")) {
        $p = Join-Path $_.FullName $f; if (Test-Path $p) { Copy-Item $p $dst -Force }
    }
}
Get-Content "$Improved\results\a3\summary.md"
