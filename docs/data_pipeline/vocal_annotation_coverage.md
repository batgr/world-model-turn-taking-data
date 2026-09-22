# Vocal annotation coverage audit

## Scope

This audit measures whether wearer/focal voice annotations cover acoustically
detected vocal activity before any `ONSET`, `OFFSET` or `NO_EVENT` actions are
constructed. It is detection/QC only: it does not modify annotations or media,
does not create pseudo-labels, and does not construct actions.

The audit keeps three concepts separate:

1. `detected_acoustic_voice_duration_s` is mixed-audio activity detected by
   Silero VAD.
2. focal voice candidates exist only when a dataset-supported identity method
   attributes the detected interval to the wearer.
3. annotation coverage is temporal overlap between those candidates and the
   source focal annotation.

Mixed-audio speech outside an annotation is never by itself evidence of a
missing wearer annotation. Such intervals remain `unresolved_identity`.

## Dataset-specific attribution

Ego4D uses its per-person voice segments to identify already annotated focal
and other-speaker activity. The available mixed audio provides no independent
identity evidence for an unannotated interval. Consequently, the audit reports
acoustic activity and unresolved duration, while focal-specific completeness is
`unresolved` rather than spuriously reported as complete.

EgoCom additionally evaluates a fixed relative-energy rule across synchronized
participant devices. The focal microphone must exceed the loudest synchronized
other microphone by the configured dB margin. This is not treated as truth: for
each recording, the rule must first reach the configured apparent precision on
both available focal and other-speaker annotation intervals. A decision below
that gate, within the energy margin, without a synchronized view, or with weak
cross-device correlation remains unresolved. The validation is in-sample
diagnostic evidence for a fixed rule, not an independent diarization benchmark.

## Time and overlap

Native audio PTS remains the media-time authority. FFmpeg decodes each requested
window with asynchronous resampling so timestamp gaps/overlaps are represented
in the decoded timeline; the explicit media-to-canonical offset maps samples to
the Ego4D clip or EgoCom conversation-part annotation timeline. Long Ego4D audio
is never aligned by naïvely concatenating PCM and using
`sample_index / sample_rate`.

Coverage and alignment are separate. Coverage uses the configured annotation
dilation and overlap thresholds. Exact overlap and signed onset/offset
differences are retained separately so small boundary shifts do not become
false omissions.

All detector, identity, overlap, merge and duration-bucket parameters live in
`conf/vocal_annotation_coverage.yaml` and are copied into the report.

## Run and outputs

Install the optional maintained VAD dependencies and run one or both corpora:

```bash
uv sync --extra vad
uv run conv-wm audit vocal-annotation-coverage --dataset egocom
uv run conv-wm audit vocal-annotation-coverage --dataset ego4d
uv run conv-wm audit vocal-annotation-coverage --dataset all
```

The configured reports root receives:

- `vocal_annotation_coverage/summary.parquet`: one row per recording/wearer;
- `vocal_annotation_coverage/uncovered_segments.parquet`: partially covered,
  uncovered and unresolved acoustic candidates (the status controls the
  interpretation);
- `vocal_annotation_coverage/report.json`: population distributions, corpus
  breakdowns, representative examples, exact configuration, detector/model
  version, Git state, command and input checksums.

No PASS/WARN/FAIL threshold is applied. The report first exposes the coverage,
duration and unresolved-identity distributions so a later decision can justify
its policy.
