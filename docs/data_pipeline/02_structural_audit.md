# Phase A2: structural audit

## Objective

A2 verifies that annotation tables have the expected shape and relationships before
later processing. It checks enforceable structural contracts; it does not assess
whether media streams or annotations are correctly aligned in time.

Profiling and validation serve different purposes:

- **Profiling is descriptive observation.** `skimpy` is available for lightweight,
  human-led inspection of distributions, nulls, and cardinalities.
- **Schema validation is an enforceable contract.** Pandera is the validation engine
  because it provides readable dataframe schemas and lazy, diagnostic failures.

## Inputs and checks

The audit reads the raw EgoCom `video_info.csv` and
`ground_truth_transcriptions.csv` tables, plus these Ego4D interim tables:

- `clips_clean`
- `persons_clean`
- `tracking_paths_clean`
- `tracks_clean`
- `voice_segments_clean`
- `transcriptions_clean`
- `social_segments_talking_clean`
- `social_segments_looking_clean`

Pandera checks exact columns, logical dtypes, nullability, the documented EgoCom
train/validation/test split rule, Ego4D split values, and defensible unique keys.
It also checks positive EgoCom speaker counts and durations, positive tracking-box
widths and heights, and non-negative unmapped-frame counts.

The project-specific relation checks verify:

- EgoCom ground-truth conversation IDs reference `video_info`.
- Every Ego4D child-table `clip_uid` references `clips_clean`.
- Tracking-path and voice-segment person keys reference `persons_clean`.
- Track keys reference `tracking_paths_clean`.

Transcription and social-segment speaker fields are not person foreign keys because
the source uses `-1` for an unknown speaker. Social `target` semantics are also left
unconstrained rather than inferred from the current snapshot.

## Run and output

From the repository root:

```bash
uv run python -m conv_wm.data.audits.run_structural
```

The command prints a concise PASS/FAIL summary and writes the machine-readable
report to `${paths.reports}/structural/structural_audit.json`.

## Deliberate exclusions and limitations

A2 does not check FPS, frame or audio timestamps, ordering of start/end fields,
seconds-to-frame consistency, A/V synchronization, drift, or resampling. It also
does not judge transcript content or scientific suitability. An empty table can
therefore pass when its columns and dtypes satisfy its contract; this currently
applies to `social_segments_looking_clean`.

The next stage is phase B, the temporal/media audit.
