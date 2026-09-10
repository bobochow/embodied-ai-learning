# Paper Radar Automation

This pipeline discovers recent embodied-AI papers from two external indexes,
normalizes their arXiv identifiers, re-fetches canonical arXiv metadata, applies
transparent relevance scoring, and prepends a reviewable daily section to
`docs/papers/latest.md`.

## Source roles

- `Ed1sonChen/DailyArxiv`: high-recall daily discovery for VLA and WAM terms.
- `DravenALG/awesome-vla-wam`: curated taxonomy and a high-confidence source boost.
- arXiv API: canonical title, abstract, authors, categories, and dates.

The external repositories are discovery sources only. Their code and prose are
not copied into this repository.

## Local use

```bash
python3 scripts/papers/update.py --dry-run --lookback-days 7
python3 scripts/papers/update.py
python3 scripts/papers/update.py --check
python3 -m unittest discover -s tests -p 'test_*.py'
```

The first successful write bootstraps `data/papers/index.json` from the existing
paper archive. Each update also writes an auditable run report to
`data/papers/runs/YYYY-MM-DD.json`, including the Top 10, Top 3, and Top 1 per
track.

## Optional Chinese summaries

Without `ANTHROPIC_API_KEY`, the job emits a clearly marked extractive fallback
and opens a draft pull request for human editing. To generate Chinese summaries,
configure these repository secrets or variables:

- Secret: `ANTHROPIC_API_KEY`
- Variable or environment value: `PAPER_LLM_MODEL` (defaults to
  `claude-sonnet-4-6`)

All model output remains review-only until the draft pull request is merged.

## Failure behavior

- Empty or unavailable sources do not overwrite `latest.md`.
- Existing arXiv IDs are never inserted again.
- Generic non-robot world-model results are penalized.
- Generated HTML is escaped before insertion.
- The displayed count, first date, unique IDs, tests, and VitePress build must
  all pass before a pull request is created.
