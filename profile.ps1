function autoclean {
    param(
        [Parameter(Mandatory=$true)]
        [string]$DataPath,
        [Parameter(Mandatory=$true)]
        [string]$Task,
        [Parameter(Mandatory=$true)]
        [string]$ConfigPath,
        [Parameter(Mandatory=$false)]
        [string]$OutputPath = ".\output"
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
    
    # Create output directory if it doesn't exist
    if (-not (Test-Path $OutputPath)) {
        New-Item -ItemType Directory -Path $OutputPath | Out-Null
    }
    
    Write-Host "Using data from: $DataPath"
    Write-Host "Using configs from: $ConfigPath"
    Write-Host "Task: $Task"
    Write-Host "Output will be written to: $OutputPath"

    $ConfigFile = (Split-Path $ConfigPath -Leaf)
    
    # Set environment variables for docker-compose
    $env:EEG_DATA_PATH = $DataPath
    $env:CONFIG_PATH = (Split-Path $ConfigPath -Parent)
    $env:OUTPUT_PATH = $OutputPath
    
    # Run using docker-compose with just the task argument
    docker-compose run --rm autoclean --task $Task --data $DataPath --config $ConfigFile --output $OutputPath
} 