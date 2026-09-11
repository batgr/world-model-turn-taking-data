## Development

### Environment

Install the project and its dependencies with:

```bash
uv sync
```

### Structural data audit

Run phase A2 against the configured EgoCom and Ego4D data:

```bash
uv run python -m conv_wm.data.audits.run_structural
```

The JSON report is written to
`${paths.reports}/structural/structural_audit.json`. See
[`docs/data_pipeline/02_structural_audit.md`](docs/data_pipeline/02_structural_audit.md)
for the checks and deliberate exclusions.
