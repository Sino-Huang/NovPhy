# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- Create: `gh issue create --title "..." --body "..."`
- Read: `gh issue view <number> --comments`
- List: `gh issue list` with appropriate state and label filters
- Comment: `gh issue comment <number> --body "..."`
- Label: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`
- Close: `gh issue close <number> --comment "..."`
- Infer the repository from the Git remote.
- Pull requests are not part of the triage request surface by default.

## Skill operations

- Publishing a ticket means creating a GitHub issue.
- Fetching a ticket means running `gh issue view <number> --comments`.
- Wayfinder maps and children use GitHub issues, sub-issues when available, native dependencies when available, and documented task-list/`Blocked by:` fallbacks otherwise.
