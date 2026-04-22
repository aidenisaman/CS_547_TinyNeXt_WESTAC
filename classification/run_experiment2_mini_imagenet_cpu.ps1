param(
    [string]$RepoRoot = "",
    [string]$ImageRoot = "",
    [string]$TrainCsv = "",
    [string]$ValCsv = "",
    [string]$TestCsv = "",
    [string]$PreparedDataRoot = "",
    [string]$Model = "tinynext_t",
    [string]$Device = "cpu",
    [int]$NumWorkers = 2,
    [int]$BatchSize = 32,
    [int]$InputSize = 224,
    [int]$SmokeEpochs = 2,
    [int]$FullEpochs = 20,
    [int]$ExpectedClasses = 100,
    [double]$TrainRatio = 0.8,
    [int]$SplitSeed = 42,
    [string]$OutputRoot = "logs_exp2_mini_imagenet_cpu",
    [string]$PythonExe = "python",
    [switch]$RunBaseline,
    [switch]$RunTuned,
    [switch]$PrepareOnly,
    [switch]$SmokeOnly,
    [switch]$RunFull,
    [switch]$ForceRebuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
if ([string]::IsNullOrWhiteSpace($ImageRoot)) {
    $ImageRoot = $RepoRoot
}
if ([string]::IsNullOrWhiteSpace($TrainCsv)) {
    $TrainCsv = Join-Path $RepoRoot "train.csv"
}
if ([string]::IsNullOrWhiteSpace($ValCsv)) {
    $ValCsv = Join-Path $RepoRoot "val.csv"
}
if ([string]::IsNullOrWhiteSpace($TestCsv)) {
    $TestCsv = Join-Path $RepoRoot "test.csv"
}
if ([string]::IsNullOrWhiteSpace($PreparedDataRoot)) {
    $PreparedDataRoot = Join-Path $RepoRoot "Data\mini_imagenet_100_folder"
}

if (-not $RunBaseline -and -not $RunTuned) {
    $RunBaseline = $true
    $RunTuned = $true
}

$prepareScript = Join-Path $PSScriptRoot "prepare_mini_imagenet_folder.py"
$mainScript = Join-Path $PSScriptRoot "main.py"
Set-Location $RepoRoot
$env:PYTHONPATH = "$RepoRoot;$PSScriptRoot"

$prepArgs = @(
    $prepareScript,
    "--image-root", $ImageRoot,
    "--train-csv", $TrainCsv,
    "--val-csv", $ValCsv,
    "--test-csv", $TestCsv,
    "--output-root", $PreparedDataRoot,
    "--merge-csv-splits",
    "--train-ratio", $TrainRatio.ToString(),
    "--seed", $SplitSeed.ToString(),
    "--mode", "hardlink",
    "--expected-classes", $ExpectedClasses.ToString()
)
if ($ForceRebuild) {
    $prepArgs += "--clean"
}

Write-Host "Preparing mini-ImageNet ImageFolder data at: $PreparedDataRoot"
& $PythonExe @prepArgs
if ($LASTEXITCODE -ne 0) {
    throw "Dataset preparation failed with exit code $LASTEXITCODE"
}

if ($PrepareOnly) {
    Write-Host "Prepare-only mode complete."
    exit 0
}

function Invoke-Run {
    param(
        [string]$RunName,
        [string]$StageName,
        [int]$Epochs,
        [string[]]$ExtraArgs
    )

    $outputDir = "$OutputRoot/$StageName/$RunName"
    $args = @(
        $mainScript,
        "--model", $Model,
        "--data-set", "FOLDER",
        "--data-path", $PreparedDataRoot,
        "--train-split", "train",
        "--val-split", "val",
        "--num-classes", "100",
        "--input-size", $InputSize.ToString(),
        "--batch-size", $BatchSize.ToString(),
        "--epochs", $Epochs.ToString(),
        "--num_workers", $NumWorkers.ToString(),
        "--device", $Device,
        "--output_dir", $outputDir
    ) + $ExtraArgs

    Write-Host "Starting $StageName run: $RunName"
    & $PythonExe @args
    if ($LASTEXITCODE -ne 0) {
        throw "$StageName run '$RunName' failed with exit code $LASTEXITCODE"
    }
}

$baselineArgs = @(
    "--reprob", "0.0",
    "--mixup", "0.0",
    "--cutmix", "0.0"
)

$tunedArgs = @(
    "--reprob", "0.1",
    "--aa", "rand-m9-mstd0.5-inc1",
    "--mixup", "0.2",
    "--cutmix", "0.2",
    "--weight-decay", "0.05",
    "--lr", "0.004",
    "--warmup-epochs", "10"
)

if ($RunBaseline) {
    Invoke-Run -RunName "baseline" -StageName "smoke" -Epochs $SmokeEpochs -ExtraArgs $baselineArgs
}
if ($RunTuned) {
    Invoke-Run -RunName "tuned" -StageName "smoke" -Epochs $SmokeEpochs -ExtraArgs $tunedArgs
}

if (-not $SmokeOnly -and $RunFull) {
    if ($RunBaseline) {
        Invoke-Run -RunName "baseline" -StageName "full" -Epochs $FullEpochs -ExtraArgs $baselineArgs
    }
    if ($RunTuned) {
        Invoke-Run -RunName "tuned" -StageName "full" -Epochs $FullEpochs -ExtraArgs $tunedArgs
    }
}

Write-Host "Experiment 2 mini-ImageNet workflow complete."
Write-Host "Logs are under: $OutputRoot/<smoke|full>/<baseline|tuned>/<model>/<timestamp>/rank0.log"
