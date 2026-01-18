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