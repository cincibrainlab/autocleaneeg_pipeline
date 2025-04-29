Autoclean with Docker
======================

This tutorial explains how to use Autoclean with Docker.

Prerequisites
-------------

*   Docker installed on your machine
*   Autoclean repository cloned on your machine

Steps to use Autoclean with Docker
---------------------------------

1.  Clone the Autoclean repository

    .. code-block:: bash

       git clone https://github.com/autoclean-io/autoclean.git

2.  Build the Docker image

    .. code-block:: bash

       docker build -t autoclean .

3.  Add the autoclean command to your shell profile

    **Windows**

    .. code-block:: bash

        #a) Copy this file to your PowerShell profile:
        Copy-Item profile.ps1 $PROFILE

        #OR

        #b) Add this line to your existing profile:
        . .\profile.ps1

    **Linux**

    .. code-block:: bash

        #Copy the bash script to the /usr/local/bin directory

        sudo cp autoclean.sh /usr/local/bin/autoclean

        #Make the autoclean command executable
        
        chmod +x /usr/local/bin/autoclean

4.  Run 'autoclean' from the command line

    .. code-block:: bash

       autoclean -DataPath "C:\Data\raw" -Task "RestingEyesOpen" -ConfigPath "C:\configs\autoclean_config.yaml"

