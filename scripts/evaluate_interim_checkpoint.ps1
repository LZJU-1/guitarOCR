param(
    [int]$TargetStep = 12000,
    [int]$MaxSamples = 90,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$OutputRoot = Join-Path $ProjectRoot "output\glm_ocr_measure_sequence_v2_lora"
$Checkpoint = Join-Path $OutputRoot "checkpoint-$TargetStep"
$Manifest = Join-Path $ProjectRoot "database\gp8_measure_sequence_v2\manifests\test.jsonl"
$StatusPath = Join-Path $ProjectRoot "reports\checkpoint${TargetStep}_interim_status.json"
$CorrectPredictions = Join-Path $ProjectRoot "reports\checkpoint${TargetStep}_correct${MaxSamples}_predictions.jsonl"
$CorrectMetrics = Join-Path $ProjectRoot "reports\checkpoint${TargetStep}_correct${MaxSamples}_metrics.json"
$ShuffledPredictions = Join-Path $ProjectRoot "reports\checkpoint${TargetStep}_shuffled${MaxSamples}_predictions.jsonl"
$ShuffledMetrics = Join-Path $ProjectRoot "reports\checkpoint${TargetStep}_shuffled${MaxSamples}_metrics.json"

function Write-InterimStatus {
    param(
        [string]$Stage,
        [string]$Message = "",
        [int]$ResumeProcessId = 0
    )
    [ordered]@{
        target_step = $TargetStep
        max_samples = $MaxSamples
        stage = $Stage
        message = $Message
        resume_process_id = $ResumeProcessId
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Test-CheckpointComplete {
    $Adapter = Join-Path $Checkpoint "adapter_model.safetensors"
    $State = Join-Path $Checkpoint "trainer_state.json"
    if (-not (Test-Path -LiteralPath $Adapter -PathType Leaf) -or
        -not (Test-Path -LiteralPath $State -PathType Leaf)) {
        return $false
    }
    try {
        $TrainerState = Get-Content -LiteralPath $State -Raw | ConvertFrom-Json
        return [int]$TrainerState.global_step -eq $TargetStep -and
            (Get-Item -LiteralPath $Adapter).Length -gt 30000000
    }
    catch {
        return $false
    }
}

function Invoke-CheckpointInference {
    param(
        [ValidateSet("none", "shuffled")]
        [string]$ImageAblation,
        [string]$Predictions,
        [string]$Metrics
    )
    $Arguments = @(
        "-m", "guitarocr.evaluation.infer_glm_ocr_measure_sequence",
        "--manifest", $Manifest,
        "--adapter", $Checkpoint,
        "--predictions", $Predictions,
        "--metrics", $Metrics,
        "--max-samples", $MaxSamples,
        "--maximum-attempts", 1,
        "--seed", 20260715,
        "--image-ablation", $ImageAblation
    )
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Checkpoint $TargetStep inference ($ImageAblation) failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    throw "The source-disjoint test manifest is missing: $Manifest"
}

Write-InterimStatus -Stage "waiting_for_checkpoint"
while (-not (Test-CheckpointComplete)) {
    Start-Sleep -Seconds 2
}

# Let all checkpoint files finish flushing before interrupting the live trainer.
$InitialAdapterSize = (Get-Item -LiteralPath (Join-Path $Checkpoint "adapter_model.safetensors")).Length
Start-Sleep -Seconds 10
$FinalAdapterSize = (Get-Item -LiteralPath (Join-Path $Checkpoint "adapter_model.safetensors")).Length
if ($InitialAdapterSize -ne $FinalAdapterSize -or -not (Test-CheckpointComplete)) {
    throw "Checkpoint $TargetStep did not remain stable after it appeared."
}

Write-InterimStatus -Stage "pausing_training"
$TrainingProcesses = Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and $_.CommandLine -and (
        ($_.CommandLine -match "llamafactory-cli" -and
            $_.CommandLine -match "train .*glm_ocr_measure_sequence_v2") -or
        $_.CommandLine -match "train_glm_ocr_measure_sequence.ps1"
    )
}
$TrainingProcesses |
    Sort-Object @{ Expression = { if ($_.Name -eq "python.exe") { 0 } elseif ($_.Name -like "llamafactory*") { 1 } else { 2 } } } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
Start-Sleep -Seconds 8

$EvaluationError = $null
Push-Location $ProjectRoot
try {
    $env:PYTHONUTF8 = "1"
    $env:PYTHONPATH = $ProjectRoot
    $env:CUDA_VISIBLE_DEVICES = "0"
    Write-InterimStatus -Stage "evaluating_correct_images"
    Invoke-CheckpointInference -ImageAblation "none" `
        -Predictions $CorrectPredictions -Metrics $CorrectMetrics
    Write-InterimStatus -Stage "evaluating_shuffled_images"
    Invoke-CheckpointInference -ImageAblation "shuffled" `
        -Predictions $ShuffledPredictions -Metrics $ShuffledMetrics
}
catch {
    $EvaluationError = $_.Exception.Message
}
finally {
    Pop-Location
}

$TrainScript = Join-Path $PSScriptRoot "train_glm_ocr_measure_sequence.ps1"
$ResumeStdout = Join-Path $ProjectRoot "output\glm_ocr_measure_sequence_v2_resume${TargetStep}.stdout.log"
$ResumeStderr = Join-Path $ProjectRoot "output\glm_ocr_measure_sequence_v2_resume${TargetStep}.stderr.log"
$ResumeArguments = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $TrainScript,
    "-ResumeFromCheckpoint", $Checkpoint
)
$ResumeProcess = Start-Process -FilePath "powershell.exe" -ArgumentList $ResumeArguments `
    -WindowStyle Hidden -RedirectStandardOutput $ResumeStdout `
    -RedirectStandardError $ResumeStderr -PassThru

if ($EvaluationError) {
    Write-InterimStatus -Stage "evaluation_failed_training_resumed" `
        -Message $EvaluationError -ResumeProcessId $ResumeProcess.Id
    throw $EvaluationError
}

Write-InterimStatus -Stage "complete_training_resumed" `
    -ResumeProcessId $ResumeProcess.Id
