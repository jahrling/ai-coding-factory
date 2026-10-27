# CLAUDE.md

Guidance for Claude Code when working in this repository. See `AGENTS.md` for
the detailed architecture, agents, skills, and conventions.

## Development workflow (required)

- **Never push feature work directly to the default branch (`master`).** Create
  a feature branch for every change.
- **Open a pull request early — a draft PR is fine — and push work as small,
  incremental commits**, so the changes can be reviewed step-by-step in the PR
  as they evolve, rather than landing as one large commit at the end.
- Prefer several focused commits over one squashed drop; keep **unrelated
  changes in separate branches/PRs**.
- **Wait for my review.** Don't merge PRs yourself unless I explicitly ask.
- If a change was already pushed to `master` by mistake, don't force-push to
  rewrite shared history without asking first.
