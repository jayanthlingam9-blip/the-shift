# Back up the local Docker Postgres (ragdb) to a compressed dump.
# Runs pg_dump INSIDE the container, so no Postgres client is needed on Windows.
#
#   powershell -File scripts/backup_db.ps1
#
# Output: backups/ragdb_<timestamp>.dump  (custom format, restorable with restore_db.ps1)
#
# NOTE: the dump is written to a file inside the container and copied out with
# `docker cp`. We deliberately AVOID PowerShell's `>` redirection, which re-encodes
# binary output as UTF-16 and corrupts the dump.

$ErrorActionPreference = "Stop"

$container = "theshift_postgres"
$user      = "raguser"
$db        = "ragdb"

$root      = Split-Path -Parent $PSScriptRoot
$backupDir = Join-Path $root "backups"
if (-not (Test-Path $backupDir)) { New-Item -ItemType Directory -Path $backupDir | Out-Null }

$stamp     = Get-Date -Format "yyyy-MM-dd_HHmmss"
$fileName  = "ragdb_$stamp.dump"
$out       = Join-Path $backupDir $fileName
$inDocker  = "/tmp/$fileName"

Write-Host "Dumping $db from container $container ..."
# -Fc = custom compressed format (includes vector data + indexes). Write to a
# file inside the container, then copy it to the host (binary-safe).
docker exec $container pg_dump -U $user -Fc -f $inDocker $db
if ($LASTEXITCODE -ne 0) { throw "pg_dump failed (exit $LASTEXITCODE)" }

docker cp "${container}:$inDocker" $out
if ($LASTEXITCODE -ne 0) { throw "docker cp failed (exit $LASTEXITCODE)" }

docker exec $container rm -f $inDocker | Out-Null

$size = "{0:N2} MB" -f ((Get-Item $out).Length / 1MB)
Write-Host "Backup written: $out ($size)"
