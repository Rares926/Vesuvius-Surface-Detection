# How we work in this project

This file lists the working conventions for Claude sessions in this folder. It grows over time as we add new conventions.

## Reference docs

- [planner.md](planner.md): how we plan the work. It holds the task groups, the tasks, their status and the files each
  task produced, along with the rules for updating it. Check it at the start of a session and keep it current while
  working.
- [.claude/context/kaggle_info.md](.claude/context/kaggle_info.md): the competition brief, covering the task, data
  layout, metric, submission constraints and modelling notes. Read it before any modelling, data or submission work.
- `assignment.md`: my interview assignment. It stays outside `.claude/`.
- [.claude/context/session_handoff.md](.claude/context/session_handoff.md): the state of the project and the
  decisions from the first long session (up to 2026-11-25). Read it at the start of a new session.

## Environment

- Python runs in the conda environment `vesuvius` (`conda run -n vesuvius python ...`). Do not create venvs.
- Install packages into it only when I ask.

## Where files go

All files Claude writes for me to read go under `.claude/`, in one of these two folders:

- `.claude/context/`: knowledge we build up step by step from my requests and our discussions.
  - Files that apply to the whole project, such as `kaggle_info.md`, sit at the top level of `context/`.
  - Each planner task gets a folder `context/<task>/`, where `<task>` is the task ID plus a short name (e.g.
    `U-1_understand_the_data`).
    - Big md files go in `context/<task>/context/`.
    - Diagrams go in `context/<task>/diagram/`.
- `.claude/reports/`: research reports, written only when I ask for one. I say what to research and where to save it.
  We then use the report as context while working on tasks.

## Code

- All project code goes in `src/`, one subfolder per component (e.g. `src/view/`). This follows the layout the
  competition brief suggests.
- Planner tasks in the Implement group produce their code here.

Other rules:

- Code, data and other project files do not go in `.claude/`.
- `.claude/` is also where Claude Code keeps its own config, such as `settings.json`, `skills/` and `agents/`. Leave those
  files alone unless asked.
