# Structural validation

## Objective

Structural validation verifies that annotation tables have the expected shape and
relationships before later processing. It checks enforceable structural contracts; it does not assess
whether media streams or annotations are correctly aligned in time.

Profiling and validation serve different purposes:

- **Profiling is descriptive observation.** `skimpy` is available for lightweight,
  human-led inspection of distributions, nulls, and cardinalities.
- **Schema validation is an enforceable contract.** Pandera is the validation engine
  because it provides readable dataframe schemas and lazy, diagnostic failures.

## Inputs and checks

Each registered dataset declares its tables and relations in a
`StructuralSpec` (`conv_wm/data/datasets/<name>.py`); the generic orchestration
(`conv_wm.data.audits.structural`) validates whatever is declared. EgoCom
declares the raw `video_info.csv` and `ground_truth_transcriptions.csv` tables
plus their maintained source-faithful interim tables;
Ego4D declares these interim tables:

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

Transcription and social-segment speaker fields can use `-1` for an unknown
speaker. Annotation integrity handles those semantic references; structural
validation does not infer social `target` meaning.

## Run and output

From the repository root:

```bash
uv run conv-wm audit structure
```

The command prints a concise PASS/FAIL summary, exits with code 2 on FAIL and
writes the machine-readable report to
`${paths.reports}/structural/structural_audit.json` (`datasets.<name>.tables`
and `datasets.<name>.relations`, plus provenance).

## Deliberate exclusions and limitations

Structural validation does not check FPS, frame or audio timestamps, ordering
of start/end fields, seconds-to-frame consistency, A/V synchronization, drift,
or resampling. It also does not judge transcript content or scientific
suitability.

Temporal semantics are covered by the temporal audits
([`03_temporal_audit.md`](03_temporal_audit.md)) and by annotation integrity
([`05_annotation_audit.md`](05_annotation_audit.md)).
