param([ValidateRange(1024, 65535)][int]$Port = 8653)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $projectRoot 'runs\inhabitants'
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$listener = Get-NetTCPConnection -LocalAddress '127.0.0.1' -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)"
    if ($existing.CommandLine -match 'haishool\.inspect_world\s+serve') {
        Write-Output "http://127.0.0.1:$Port/"
        exit 0
    }
    throw "Port $Port belongs to another process; choose a different -Port."
}
$pythonPath = (Get-Command python.exe).Source
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdout = Join-Path $runRoot "server-$stamp.stdout.log"
$stderr = Join-Path $runRoot "server-$stamp.stderr.log"
$server = Start-Process -FilePath $pythonPath -ArgumentList @('-u', '-m', 'haishool.inspect_world', 'serve', '--root', 'runs/inhabitants', '--host', '127.0.0.1', '--port', "$Port") -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
@{ pid=$server.Id; url="http://127.0.0.1:$Port/"; stdout=$stdout; stderr=$stderr; root=$runRoot } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'server.json') -Encoding utf8
Write-Output "http://127.0.0.1:$Port/"
