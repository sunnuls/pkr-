# Run the phone bot in its own window and restart it if it exits.
param(
    [string]$Serial = "e2d0507",
    [switch]$Watch
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
New-Item -ItemType Directory -Force -Path "runs\live_play" | Out-Null

$botArgs = @("-u", "tools/play_phone.py", "--serial", $Serial,
             "--save-dir", "runs/live_play", "--log", "runs/live_play/bot.log")
if (-not $Watch) { $botArgs += "--tap" }

while ($true) {
    if (Test-Path "runs\live_play\STOPPED") {
        Write-Host "$(Get-Content runs\live_play\STOPPED) - delete runs\live_play\STOPPED to play again"
        [console]::beep(880, 600)
        break
    }
    python @botArgs 2>&1 | Tee-Object -FilePath "runs\live_play\console.log" -Append
    Write-Host "bot exited (code $LASTEXITCODE) - restarting in 3s, close this window to stop"
    Start-Sleep 3
}
