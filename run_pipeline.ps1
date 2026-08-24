$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$python = (Get-Command python).Source
$log = "output\pipeline.log"
New-Item -ItemType Directory -Force -Path output | Out-Null

"=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -Append -Encoding utf8 $log
& $python collect.py    2>&1 | Out-File -Append -Encoding utf8 $log
& $python summarize.py  2>&1 | Out-File -Append -Encoding utf8 $log
& $python dashboard.py  2>&1 | Out-File -Append -Encoding utf8 $log
