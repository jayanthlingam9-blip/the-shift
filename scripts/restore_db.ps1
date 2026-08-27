# Restore the local Docker Postgres (ragdb) from a dump made by backup_db.ps1.
# Copies the dump INTO the container and runs pg_restore there.
#
#   powershell -File scripts/restore_db.ps1 backups/ragdb_2026-06-15_120000.dump
#
# WARNING: --clean drops existing objects before restoring. Make a fresh backup
# first if the current data matters.
#
# NOTE: we use `docker cp` rather than piping stdin, to stay binary-safe under
# Windows PowerShell.

param(
    [Parameter(Mandatory = $true)]
    [string]$DumpFile
)

$ErrorActionPreference = "Stop"

$container = "theshift_postgres"
$user      = "raguser"
$db        = "ragdb"

if (-not (Test-Path $DumpFile)) { throw "Dump file not found: $DumpFile" }

$inDocker = "/tmp/restore_input.dump"

Write-Host "Copying dump into container ..."
docker cp $DumpFile "${container}:$inDocker"
if ($LASTEXITCODE -ne 0) { throw "docker cp failed (exit $LASTEXITCODE)" }

Write-Host "Restoring into $db (existing objects will be dropped) ..."
docker exec $container pg_restore -U $user -d $db --clean --if-exists $inDocker
if ($LASTEXITCODE -ne 0) { throw "pg_restore failed (exit $LASTEXITCODE)" }

docker exec $container rm -f $inDocker | Out-Null
Write-Host "Restore complete."
