# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

The canonical project issue tracker is `Sino-Huang/NovPhy`. Every `gh` issue operation MUST pass `--repo Sino-Huang/NovPhy`; do not infer the tracker from the upstream remote or a `gh` default repository.

## Conventions

- Create: `gh issue create --title "..." --body "..."`
- Read: `gh issue view <number> --comments`
- List: `gh issue list` with appropriate state and label filters
- Comment: `gh issue comment <number> --body "..."`
- Label: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`
- Close: `gh issue close <number> --comment "..."`
- Use `Sino-Huang/NovPhy` even when another remote, such as `upstream`, points to `phy-q/NovPhy`./tdfasd
- Pull requests are not part of the triage request surface by default.

## Skill operations

- Publishing a ticket means creating a GitHub issue.
- Fetching a ticket means running `gh issue view <number> --comments`.
- Wayfinder maps and children use GitHub issues, sub-issues when available, native dependencies when available, and documented task-list/`Blocked by:` fallbacks otherwise.
