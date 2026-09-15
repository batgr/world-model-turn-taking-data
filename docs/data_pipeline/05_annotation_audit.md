# Annotation integrity

## Objective and architecture

Annotation integrity determines whether each declared source is structurally,
temporally, relationally, and referentially usable. It detects and reports; it
does not correct. Run:

```bash
uv run conv-wm audit annotations
```

Each dataset declares typed `AnnotationSourceSpec` objects: scope, temporal
coordinates, identity, media/entity references, bounds, payload fields,
provenance, and known limitations. Generic checks cover finite and ordered
timestamps, bounds, duplicate identities, references, and declared
cross-source comparisons. Exact interval identity and directional temporal
overlap are both reported. A comparison can declaratively exclude a sentinel,
require populated payload, or apply a diagnostic overlap expansion without
changing either source.

Outputs below `${paths.reports}/annotations/` are:

- `annotation_integrity.json`, the full source contract and results;
- `annotation_sources.parquet`, `annotation_anomalies.parquet`, and
  `annotation_cross_source.parquet`, flat report tables.

Every source is kept separate. Corrections belong to
[`06_cleaning.md`](06_cleaning.md), and downstream media-domain intersection
belongs to temporal projection.

## Regenerated corpus findings

The annotation audit passes after maintained cleaning. Sources with documented
warnings remain usable with caveats.

| Dataset/source | Rows | Relevant findings |
| --- | ---: | --- |
| Ego4D clips | 439 | 10 slightly negative video starts; all media references resolve. |
| Ego4D persons | 3,231 | Camera wearer has no face track. |
| Ego4D voice | 40,383 | 260 clip-bound overshoots; five zero-duration intervals. |
| Ego4D transcriptions | 39,191 | 2,630 unknown speakers; 351 clip-bound overshoots. |
| Ego4D talking | 43,172 | 2,791 unknown speakers; two concrete source person IDs do not occur in the clip persons table; 378 clip-bound overshoots. |
| Ego4D looking | 10,211 | 61 clip-bound overshoots; current payload is uniformly target-null/false. |
| Ego4D tracking paths | 38,242 | No integrity warning. |
| Ego4D tracks | 5,523,640 | 490 boxes extend beyond the declared clip frame range. |
| EgoCom video info | 175 | Available POV metadata; every media reference resolves. |
| EgoCom ground truth | 359,536 | 177,230 fully timed rows and 343 start-only rows; nullable timing is source-defined. |

Small negative starts, overshoots, and out-of-range frames are diagnostics, not
destructive-cleaning instructions.

## Cross-source evidence

All rates below were recomputed from the regenerated Parquet files. “Left” and
“right” preserve the order shown.

| Comparison | Left rows | Right rows | Exact left / right | Overlap left / right |
| --- | ---: | ---: | ---: | ---: |
| voice ↔ transcription, person-level | 40,383 | 39,191 | 85.5% / 88.1% | 86.7% / 89.3% |
| voice ↔ known-speaker transcription | 40,383 | 36,561 | 85.5% / 94.4% | 86.7% / 95.8% |
| voice ↔ known-speaker transcription, diagnostic ±0.5 s | 40,383 | 36,561 | 85.5% / 94.4% | 87.1% / 97.1% |
| voice ↔ transcription, clip-level | 40,383 | 39,191 | 87.0% / 89.6% | 95.7% / 97.2% |
| voice ↔ all talking | 40,383 | 43,172 | 97.9% / 91.6% | 99.4% / 93.3% |
| voice ↔ talking with explicit target | 40,383 | 12,453 | 29.8% / 96.7% | 30.6% / 99.6% |

Exact identity means equal shared entity key, start, and end. Voice and
transcription are still independent annotations and should be joined by
compatible identity plus overlap, not forced timestamp equality. The relaxed
comparison is diagnostic only. Short uncovered voice intervals remain valid:
their median duration is 0.965 s, compared with 1.340 s across all voice
intervals.

The corrected talking population shows that talking is largely a re-annotation
of voice. The old 30.6% apparent talking coverage was caused by first deleting
target-null rows; it now correctly describes only the explicit-target subset.

## EgoCom participant versus POV semantics

`video_info` enumerates available recordings/camera wearers. It is not a
complete participant registry. Ground-truth vocal activity can therefore name
a participant without that participant having an own POV. The 13,512 such rows
are valid, retained, and reported only as `participant_has_own_pov=False` in
the cleaning diagnostic. They do not produce a dangling-participant anomaly.

## Unresolved source semantics

- Ego4D target null means only that an explicit target ID is unavailable; no
  stronger class is inferred.
- Two talking rows name concrete people absent from their clip person lists;
  these are warned about but preserved.
- Current looking rows are all `target=null` and `is_at_me=False`, despite the
  benchmark's looking-at-me framing. The values are preserved without an
  invented correction.
- Cross-source mismatches are evidence to retain, not rows to delete.
