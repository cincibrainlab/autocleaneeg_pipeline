pre-instruction hooks:
- please make a surgical git commit before edits
overall strategy:
- use quarto (qmd) website output
    - strategically implement all quarto elements including callouts
    - organization of pages: strategic, senior PM, scalable
    - should be unified quarto website in the plans folder.
    be familiar with the docs: https://quarto.org/docs/reference/projects/websites.html
- formating:
    - APA 7 throughout (headings, citations if present, tables/figures, tone).
    - NIH-like formatting (whitespace/margins/typography)
- use dense prose to explain reasoning and steps
- use formal tables and figures with captions
- strategically decide which belong in the main body vs. appendices
- clear section headers and strict, consistent organization
- coding:
    - explain each code listing clearly and concisely
    - prefer linear scripts over deeply nested functions
    - deliverable should be reproducible and self-contained
    - minimize and optimize file outputs to avoid directory clutter
    - group / consolidate code to avoid excessive code blocks.
- Following instructions + filenaming strategy
    - RFC-style Design Docs (numbered, reviewed, archived)
    - For each set of instructions render a pair of files in a subfolder that share the same basename.
        - Reasoning Plan (.md): Record the intended instructions, their order, and the rationale at a high level, without results or execution details; keep it short and editable to steer or revise the downstream work.
        - Executed Document (.qmd): Implement and execute the full instruction set, including code, analysis, and outputs, producing the authoritative rendered result. This is organized and integrated into the larger quarto website.
    The .md file captures thinking and order, the .qmd file captures execution and results and they are always side by side in the same place.
- Be mindful that this file will be run over and over again as ideopotent instructions
- for each instruction set be an energetic assistant and deploy subagents as necessary
- Available tools:
    - Autenticated Github CLI (version control, issues, etc.)
    - quarto CLI
    - prefer the use of trash (homebrew) rather than rm -rf

Instruction sets:


Clean up rules:
Will provide explit remove: and archive: and reactivate: blocks when needed
- Archive: move instruction set and output to archive folder
- Remove: delete instruction set and output
- Reactivate: move instruction set and output to active folder

Post Instruction hooks:
- give direct, essential feedback to user for edits needed to align the current file based on your actions.
    - obsolute or outdated instructions
    - breaking ideopotency
    - critical thinking errors
    - output should be 1-2 lines, elegant and formatted
- render quarto after every change (no warnings/errors that affect output integrity).
- make a surgical git commit of your edits
- update plans/main-plan-log.csv
    Maintain a CSV-based decision log that records what was decided, why it was reasonable, what evidence supported it, and how it was validated, across both data science and general work; include timestamps only when sequencing or causality matters.
- serve the website on localhost on port 10910 as a background process to not interfer
- specific project instructions
    if needed:
