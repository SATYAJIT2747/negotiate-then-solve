# Run the web app as the team's shared server: one copy, all data in this machine's runs/.
# Uses the fine-tuned model on the GPU server when it answers, the offline parser otherwise.
#
#   powershell -ExecutionPolicy Bypass -File scripts\host.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\host.ps1 -Offline     # never use the GPU
#   powershell -ExecutionPolicy Bypass -File scripts\host.ps1 -Adapters "" # every task on -Model, as before
#
#   powershell -ExecutionPolicy Bypass -File scripts\host.ps1 -Adapters adapters-router.toml  # a router server
#
# By default each task goes to its own LoRA adapter on the GPU server's base model (adapters-lora.toml, what
# the server at 10.1.25.191 runs): requests are read by nts-intents-v6-read100, which the app then asks for
# intents (NTS_INTENTS); policy, replies and explanations by nts-read100; timetable questions by nts-lookup-v1.
param(
    [string]$GpuUrl = "http://10.1.25.191:8080/v1",
    [string]$Adapters = "adapters-lora.toml",
    [string]$Model = "Qwen3.5-4B-Finetune-NTS-Read100-Q8_0",
    [int]$Port = 8000,
    [switch]$Offline
)

Set-Location (Split-Path $PSScriptRoot -Parent)

$env:NTS_HOST = "0.0.0.0"  # reachable from other machines, not only this one
$env:NTS_PORT = "$Port"
$env:NTS_LOCAL_URL = $GpuUrl
$env:NTS_LOCAL_MODEL = $Model
$needed = @($Model)
if ($Adapters) {
    $env:NTS_ADAPTERS = $Adapters
    $env:NTS_INTENTS = "1"
    # the names under [adapters], quoted ("Qwen3.5-...") or bare (nts-read100)
    $section = ""
    $needed = foreach ($line in Get-Content $Adapters) {
        if ($line -match '^\s*\[([^\]]+)\]') { $section = $Matches[1]; continue }
        if ($section -eq "adapters" -and $line -match '^\s*"?([^"=\s]+)"?\s*=') { $Matches[1] }
    }
} else {
    Remove-Item Env:NTS_ADAPTERS, Env:NTS_INTENTS -ErrorAction SilentlyContinue
}

# The app uses the model whenever it answers and the rule-based stand-ins whenever it does not, switching
# back by itself (src/web/fallback.py), so it starts on the model even when the GPU is down right now.
$parser = if ($Offline) { "offline" } else { "local" }
if (-not $Offline) {
    try {
        $models = Invoke-RestMethod -Uri "$GpuUrl/models" -TimeoutSec 5
        # what the server serves: its models (router mode) and its LoRA adapters, by file name without .gguf
        $served = @($models.data.id | ForEach-Object { [IO.Path]::GetFileNameWithoutExtension($_) })
        try {
            $root = $GpuUrl -replace '/v1/?$', ''
            $loras = Invoke-RestMethod -Uri "$root/lora-adapters" -TimeoutSec 5
            $served += @($loras | ForEach-Object { [IO.Path]::GetFileNameWithoutExtension($_.path) })
        } catch {}
        foreach ($m in $needed) {
            if (-not ($served -contains $m)) {
                Write-Host "GPU server is up but does not serve $m; it serves: $($served -join ', ')"
            }
        }
    } catch {
        Write-Host "GPU server not reachable at $GpuUrl right now; the rule-based stand-ins answer until it is back."
    }
}

Write-Host "Building the front end..."
Push-Location frontend
npm run build --silent
Pop-Location

$ips = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.|172\.)' } | Select-Object -ExpandProperty IPAddress
Write-Host ""
Write-Host "Parser: $parser"
foreach ($ip in $ips) { Write-Host "Team link: http://${ip}:$Port" }
Write-Host ""

uv run nts-web --parser $parser
