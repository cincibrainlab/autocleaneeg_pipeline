surgical integration of automation to autocleaneeg-pipeline
preparation step
the new command will be autocleaneeg serve
however, currently when repeatedly running the pipeline the pipeline will backup the output folder
this prevents automation since everytime a file is run it will end up going to an output folder with just 1 file
for automation we need to keep adding to the same folder
investigate the most surgical way to toggle this when we are running automation mode
examine downstream artifacts and outputs how many would be affected by this
ultimately the pipeline has to become ideopotent
we have build a robust code map here  docs/CODEBASE_MAP.md  for your reference and planning
write out a careful analysis, plan, and discussion for this first important step

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

