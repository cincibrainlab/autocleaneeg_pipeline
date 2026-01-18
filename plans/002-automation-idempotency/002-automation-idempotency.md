# RFC 002 Reasoning Plan: Automation Idempotency Preparation

## Intent

Document a careful analysis, plan, and discussion for enabling automation runs to reuse a single output folder without triggering automatic backups, while integrating the new serve-workspace architecture with test/live runtimes and configuration governance.

## Ordered Steps

1. Review current output directory backup behavior in `src/autoclean/utils/file_system.py` and its orchestration in `src/autoclean/core/pipeline.py`.
2. Map the proposed `autocleaneeg-pipeline serve` workspace layout, including `runtimes/test`, `runtimes/live`, and named task workspaces.
3. Identify configuration knobs (e.g., `workspace.auto_backup`) plus serve-specific YAML controls and validation handoffs.
4. Enumerate downstream artifacts and metadata that depend on backup behavior, noting overwrite risk under automation.
5. Draft an APA-style analysis and a first-step plan for a minimal toggle that preserves auditability, uptime, and configuration safety.


updates to plan:
I want to add more details on how I envision this system
autocleaneeg-pipeline serve acts as the manager of the automation
the first step is to define an autocleaneeg-pipeline serve workspace which is distinct workspace folder
this folder is specially treated since it is self contained
one of the key principals is that there will be test and live separation within this workspace
separate autoclean pipeline runtimes will be installed under runtimes/test and runtimes/live
similarly the main config file will be serve-test.yaml and serve-live.yaml
each traditional workspace folder will live in this parent folder but be specifically named
taskfile-montage-version. if you have a taskfile and a montage that's all you really need to setup   the automation

The autoclean serve command requires the path to either an existing autocleaneeg-serve folder or can create new one
then the commands will make sense 
the autocleaneeg-pipeline serve <cmd> family includes
    workspace - set or edit current autocleaneeg-pipeline serve workspace directory
    list - what automations are configured in the directory (test and live clearly labeled)
    validate - test the config file with a dry run (live or test)
    deploy  - deploy (live or test)

since we want to have 100% uptime the serve-test.yaml and serve-live.yaml sit in the home directory
the user can edit these, but there is a hidden set of yaml files that the automations run on
so prior to replacing them the autocleaneeg-pipeline serve command must validate those rigorously and then replace the deployed versions
everything should be easy and intuitive in terms of controls

but the yaml file is wonderful because we have some advantes for ecamples
the yaml file
defines the active runtime (the autocleaneeg exec)
defines what the ingestion folders (so we don't ahve to scan incomplete folders)
defines the taskfile montage combination to define the automation workspace folders etc
and as you think about it whatever else would be helpful

please incorporate thisinto your current planning for 002


## Rationale

The instruction set is preparatory, so the output focuses on analysis and planning rather than implementation. The reasoning sequence ensures the proposal is grounded in current behavior and highlights the risks to idempotency and data provenance before any code changes are considered.
