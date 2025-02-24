function autoclean {
    param(
        [Parameter(Mandatory=$true)]
        [string]$DataPath,
        [Parameter(Mandatory=$true)]
        [string]$Task,
        [Parameter(Mandatory=$false)]
        [string]$ConfigPath = ".\configs"  # Default to local configs directory
    )
    
    # Ensure paths exist
    if (-not (Test-Path $DataPath)) {
        Write-Error "Data path does not exist: $DataPath"
        return
    }
    if (-not (Test-Path $ConfigPath)) {
        Write-Error "Config path does not exist: $ConfigPath"
        return
    }
    
    Write-Host "Using data from: $DataPath"
    Write-Host "Using configs from: $ConfigPath"
    Write-Host "Task: $Task"
    
    $cmd = "docker run -it --rm" + 
           " -v `"$DataPath`":/data" +
           " -v `"$ConfigPath`":/app/configs" +
           " autoclean:latest --task $Task"
           
    Invoke-Expression $cmd
} 