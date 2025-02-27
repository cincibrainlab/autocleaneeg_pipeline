# =============================================================================
# AutoClean EEG Processing Pipeline PowerShell Script
# =============================================================================
#
# INSTALLATION:
# 1. Either:
#    a) Copy this file to your PowerShell profile:
#       Copy-Item profile.ps1 $PROFILE
#    OR
#    b) Add this line to your existing profile:
#       . .\profile.ps1
#
# USAGE:
#    autoclean -DataPath "C:\Data\raw" -Task "RestingEyesOpen" -ConfigPath "C:\configs\autoclean_config.yaml"
#
# VIEW HELP:
#    Get-Help autoclean -Detailed
#
# REQUIREMENTS:
# - Docker Desktop for Windows
# - PowerShell 5.1 or higher
# =============================================================================

function Get-AutocleanHelp {
    $help = @"
Starts containerized autoclean pipeline.
Usage: autoclean -DataPath <path> -Task <task> -ConfigPath <config> [-OutputPath <path>]

Required:
  -DataPath <path>    Directory containing raw EEG data or file path to single data file
  -Task <task>        Task type (Defined in src/autoclean/tasks)
  -ConfigPath <path>  Path to configuration YAML file

Optional:
  -OutputPath <path>  Output directory (default: .\output)
  -Help              Show this help message

Example:
  autoclean -DataPath "C:\Data\raw" -Task "RestingEyesOpen" -ConfigPath "C:\configs\autoclean_config.yaml"
"@
    Write-Host $help
}

function autoclean {
    <#
    .SYNOPSIS
    Starts containerized autoclean pipeline.
    .EXAMPLE
    autoclean -DataPath "C:\Data\raw" -Task "RestingEyesOpen" -ConfigPath "C:\configs\autoclean_config.yaml"
    #>
    [CmdletBinding(DefaultParameterSetName="Help")]
    param(
        [Parameter(ParameterSetName="Help")]
        [switch]$Help,

        [Parameter(Mandatory=$true, ParameterSetName="Execute")]
        [string]$DataPath,

        [Parameter(Mandatory=$true, ParameterSetName="Execute")]
        [string]$Task,

        [Parameter(Mandatory=$true, ParameterSetName="Execute")]
        [string]$ConfigPath,

        [Parameter(Mandatory=$false, ParameterSetName="Execute")]
        [string]$OutputPath = ".\output"
    )
    
    # Show help if -Help is specified or no parameters are provided
    if ($Help -or $PSCmdlet.ParameterSetName -eq "Help") {
        Get-AutocleanHelp
        return
    }
    
    # Ensure paths exist
    if (-not (Test-Path $DataPath)) {
        Write-Error "Data path does not exist: $DataPath"
        return
    }

    # Handle single file vs directory
    $DataMountPath = if (Test-Path $DataPath -PathType Leaf) {
        $DataFile = Split-Path $DataPath -Leaf
        Split-Path $DataPath -Parent
    } else {
        $DataFile = ""
        $DataPath
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

    $ConfigFile = (Split-Path $ConfigPath -Leaf)

    # Determine if DataPath is a file or directory
    if (Test-Path -Path $DataPath -PathType Leaf) {
        # If DataPath is a file, use its parent directory for mounting
        $MountPath = Split-Path -Parent $DataPath
        Write-Host "DataPath is a file. Mounting parent directory: $MountPath"
        $env:EEG_DATA_PATH = $MountPath
    } else {
        # If DataPath is a directory, use it directly
        $env:EEG_DATA_PATH = $DataPath
    }
    $env:CONFIG_PATH = (Split-Path $ConfigPath -Parent)
    $env:OUTPUT_PATH = $OutputPath
    
    # Run using docker-compose
    Write-Host "Starting docker-compose..."
    if ($DataFile) {
        # For single file
        docker-compose run --rm autoclean --task $Task --data $DataFile --config $ConfigFile --output $OutputPath
    } else {
        # For directory
        docker-compose run --rm autoclean --task $Task --data $DataPath --config $ConfigFile --output $OutputPath
    }
} 