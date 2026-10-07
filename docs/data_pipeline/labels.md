# Label sidecars

Rich, optional, versioned labels next to the action grid: who speaks, when the
floor changes, what comes next, what the native social annotations say — for
probes, analysis and future experiments — without changing the training fast
path. Every label is described once, in the registry
(`src/conv_wm/data/labels/catalog.py`); the per-label reference
[`labels_registry.md`](labels_registry.md) is generated from it
(`conv-wm labels docs`), so it cannot drift.

## Principles

**Observation is not interpretation.** "A starts speaking" is an observation;
"A overlaps B" is a derived observation; "A interrupts B" or "A competes for
the floor" are interpretations. The pipeline materializes observations and
deterministic derivations of them; interpretations (overlap function,
dialogue acts, addressee, backchannels, dominance, engagement, ...) are not
labels until a validated annotation or model exists, and are never fabricated
from timing.

**Three source kinds**, recorded on every label:

| `source_kind` | meaning | built by default |
| --- | --- | --- |
| `native_annotation` | provided by the corpus, at most resampled onto the grid | yes |
| `deterministic` | derived by a deterministic rule from canonical facts | yes (annotation-only extractors) |
| `external_model` | estimated by an external model or tool — never ground truth | only with `--external` |

**Modalities** describe the information a label carries, not the file it was
read from: EgoCom speaker activity comes from transcript timestamps but is a
vocal phenomenon (`audio`); Talking-To-Me is `[audio, video]`.

**Native data first.** Native annotations are never replaced by a model
(no diarization or ASR output stands in for EgoCom or Ego4D annotations).

## Where labels come from

```text
cleaned annotations ──> DatasetSpec.label_facts ──> canonical facts ─┐   (dataset adapter)
                                                                      │
vocal action grid (checked through its report) ───────────────────────┼─> derivations in
media manifest (checked, media extractors only) ──────────────────────┘   native time, then
                                                                          grid projection
     speech/  social/  text/  audio/  video/      <- one directory per extractor
```

The adapter (`data/datasets/<name>_labels.py`) is the only dataset-specific
code. It returns, per recording on its canonical clock: every participant's
native speech intervals (the wearer included, unresolved speech as its own
pseudo-participant), UNKNOWN regions, transcript tokens, social segments and
face tracks, plus a static set of **facts** it provides (`speech`, `words`,
`transcript`, `social.looking`, `social.talking`, `face_tracks`,
`media.audio`, `media.video`, `meta.*`). A label is supported by a dataset
exactly when the dataset provides every fact the label `requires`; no generic
module ever names a dataset (a test enforces it).

| extractor | reads | produces | default |
| --- | --- | --- | --- |
| `speech` | speech facts | instantaneous, events, onset context, overlap, timing, turns, next speaker, future, profiles, metadata | yes |
| `social` | social segments, face tracks | native LAM / TTM, face tracks | yes |
| `text` | transcript tokens | tokens, speech rate, closing punctuation, question cue, discourse markers | yes |
| `audio` | decoded audio | audio nuisance controls, ego-solo energy; Praat F0 with `--external` | on request |
| `video` | decoded video frames | video nuisance controls | on request |

## Time

| clock | used for |
| --- | --- |
| native timestamp | events and segments (`time_s`, `start_s`, `end_s`): exact, never rounded |
| subframe | `*_subframes` labels: cell `k` split into `S = subframes_per_step` parts (default 3 → 30 Hz, one per frame of a 30 fps video) |
| decision cell | grid labels: `[t_k, t_k + Δ)`, `t_k = k·Δ`, Δ = 0.1 s — the action grid's rows exactly |
| reference instant | timing, next-speaker and future labels: `r_k = t_k + Δ`, the cell end. Everything strictly before `r_k` is past; an event at exactly `r_k` is future |
| horizon window | future labels: `[r_k, r_k + h)` for each configured `h` |

Derivations run on native intervals first; the grid views are projections of
them. An interval marks every subframe it intersects with positive length; an
event is placed in the subframe containing its time, with the action grid's
nanosecond boundary tolerance (an event on a boundary belongs to the later
unit). Frame-indexed facts (Ego4D face tracks) are converted to seconds by
the adapter (`frame / fps`) and mapped the same way, so the 30 Hz → 10 Hz
conversion is deterministic: frame `f` lands in subframe `⌊f·S/(fps·Δ)⌋`.

## Missing information

`UNKNOWN ≠ SILENT`, `UNKNOWN ≠ False`, `not annotated ≠ negative`:

* a unit a required participant is UNKNOWN in (media not covering it, invalid
  clip, explicitly missing annotation) is **null**, never false;
* three-valued aggregates: "others active" is true as soon as one other
  participant speaks, false only when every other one is known silent;
* documented categorical codes, not missing values: floor holder `-1` NONE
  (nobody speaks), `-2` MULTIPLE; future next speaker `-1` NOBODY;
* point-in-time labels come with `*_valid`; a censored value (the event
  would lie beyond the recording end or an UNKNOWN region) is null and
  invalid — never zero, never "no event". `timing.observation_bounds` gives
  the censoring bound (`time_to_observation_end`);
* the floor (last unique speaker) is unknown from the recording start — a
  conversation part may begin mid-conversation — and after any UNKNOWN region,
  until someone speaks alone.

## Definitions worth knowing

* **Speech run**: maximal continuous speech of one participant (native
  intervals unioned). **Turn**: runs of one participant joined across pauses of
  at most `turn_max_internal_pause_s` (0.3 s) that are fully observed.
* **Floor holder** (instantaneous): the unique speaker of the unit.
  **Floor change / floor transfer**: a change of the *last unique speaker*
  between two known participants; its FTO is (start of the new holder's run) −
  (end of the previous holder's last run): negative = overlap, positive = gap.
* **Overlap types** (pairwise, Heldner & Edlund 2010, as in the earlier
  mpc-wm labels): a run *entering* another participant's run is
  `simultaneous_onset` when both start within `simultaneous_onset_tolerance_s`
  (one 30 Hz subframe), else `within` when it ends first (the initial speaker
  keeps the floor), else `between`. A type decided by a censored end is
  `undetermined`.
* **Silences**: `pause` (same speaker before and after) or `gap` (another).
* **Onset types** (per onset, context primitives kept alongside): `overlap`
  when another participant speaks at the onset instant; otherwise
  `after_silence` (the speaker was the last unique speaker) or
  `floor_transfer` (someone else was, zero-length gaps included);
  `undetermined` when anything unknown decides it.
* **Next speaker** targets preserve participant ids (no two-speaker
  assumption); simultaneous next onsets are a tie (invalid), never a guess.
* **Profiles** summarize a whole recording — future included — and are
  `diagnostic`: never an input for predicting inside the same recording. No
  scaler or clustering is fitted here; anything fitted on top of them must be
  fitted on the training split only.

## Storage

```text
${paths.processed}/labels/<dataset>/
  registry.json                   the registry the store was built against
  <extractor>/
    manifest.json                 deterministic: versions, config + digest, time
                                  semantics, input checksums, tables, materialized
                                  and unavailable labels (with reasons), tools
    grid.parquet                  rows = the action grid's rows, same order
    events.parquet                native-time events (onsets, offsets, floor changes, tokens)
    segments.parquet              native-time intervals (runs, turns, overlaps, silences, ...)
    participants.parquet          one row per (recording, participant): identity, profiles
    recordings.parquet            one row per recording: participants, span, metadata
${paths.reports}/labels/<dataset>/<extractor>/report.json   provenance of each build
```

Per-participant grid values are lists indexed by the recording's
`participant_ids` (wearer first, then natural id order); the loader returns
that column with any such label. Parquet row groups of 65 536 rows keep
recording and time filters selective.

## Building

```bash
uv run conv-wm build labels --dataset all                      # speech, social, text
uv run conv-wm build labels --dataset egocom --labels 'timing.*,events.*'
uv run conv-wm build labels --dataset ego4d --modalities audio  # speech + audio extractors
uv run conv-wm build labels --dataset all --extractors audio,video
uv run conv-wm build labels --dataset all --labels all --external   # + Praat F0 (needs the extra)
uv run conv-wm build all                                        # labels run last, annotation-only
```

The build unit is the extractor: selecting a label builds its extractor.
`--labels` and `--modalities` use the consumer's selection rules. Media
extractors decode only their own stream (audio never decodes video) from the
local corpus copy the media manifest points to; nothing is downloaded.
External labels need their extra (`uv sync --extra labels-audio`); a missing
one fails with the command that installs it.

A build refuses:

* an action grid its report no longer vouches for;
* interim annotation tables that changed after the grid was built;
* facts whose wearer speech differs from the native focal voice state the grid
  descends from (checked per recording);
* a media manifest built from another grid.

Every table and manifest is byte-identical across rebuilds from the same
inputs, configuration and code; writes are atomic (a directory is swapped in
only once complete).

```bash
uv run conv-wm labels status --dataset all     # exit 2 when an extractor is stale
uv run conv-wm audit labels --dataset all      # coverage + sanity, JSON and Markdown
uv run conv-wm labels list --include 'timing.*' --modalities audio
uv run conv-wm labels docs                     # regenerate labels_registry.md
```

A store is **stale** when the `labels` configuration digest, the action grid,
the registry or label schema version changed, or a table no longer matches
its manifest. The coverage audit reports, for every label and dataset,
support, build status, coverage, valid and missing shares, plus sanity
statistics (speaking and overlap shares, onset/offset counts, onset types,
floor transfers and FTO distribution, turn durations, time-to-next
distributions, labels per modality). It never modifies a label.

## Reading labels

Each extractor directory holds plain Parquet tables; `registry.json` names
each label's table and columns. Read only what you need:

```python
import pyarrow.parquet as pq

grid = pq.read_table(
    "data/processed/labels/egocom/speech/grid.parquet",
    columns=["recording_id", "decision_index", "participant_ids",
             "speaker_activity", "time_to_next_ego_onset", "time_to_next_ego_onset_valid"],
)
```

`conv-wm labels list --include "timing.*" --modalities audio` lists the
labels a selection resolves to: `include` takes exact names, `family.*` and
`all`; `modalities` keeps a label only when all of its modalities are listed.

## Adding a label or a family

1. Add the `LabelSpec` to `data/labels/catalog.py` (and the family to
   `FAMILIES` in `registry.py` if new): meaning, level, modalities, dtype and
   shape, time reference, source kind, role, validity, `requires`, storage.
2. Compute its columns in the extractor that owns it (pure functions in
   `data/labels/`), with explicit validity.
3. Bump `REGISTRY_VERSION` (and the extractor's rule version if a derivation
   changes); stores built with another version are refused as stale.
4. `conv-wm labels docs`; add unit tests on synthetic sequences.

## Licensing

Label sidecars are derived annotations: the release ships them next to the
action grid (`data/labels/` in each release directory, checked against the
grid the card vouches for) and never ships raw media. EgoCom and Ego4D
licences apply to what is derived from them. External tools: Praat /
Parselmouth are GPL-3.0 (covering the software, not its numeric outputs);
openSMILE (eGeMAPS) is not used — its licence restricts use to research and
education and needs review before redistributing derived features; model
weights (MediaPipe, MMPose, WhisperX, pyannote) are never packaged here.

## Not implemented, deliberately

* interpretations without a validated source: overlap function, dialogue
  acts, adjacency pairs, agreement, repair, relevance, common ground, semantic
  completion, addressee, backchannels, engagement, dominance and the other
  social states;
* lexical and syntactic completion (need a validated model);
* video model features — face landmarks, head pose, gaze, mouth, body and hand
  pose, gestures, geometry, camera motion (MediaPipe / MMPose): no extractor
  is implemented or validated on these egocentric frames;
* other participants' prosody (far-field on the wearer's microphone, no
  validated attribution), eGeMAPS (licence), MFCC (no use yet), voice quality
  (unreliable on these recordings), final lengthening (needs phone alignment),
  perceptual loudness;
* diarization and ASR back-ends (native annotations exist).
