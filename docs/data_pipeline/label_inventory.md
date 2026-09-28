# Label inventory for audio-only evaluation

Point-in-time inventory, 2026-09-28. The definitions and full validity semantics live in [the generated registry](labels_registry.md) (`catalog.py` is its source of truth). The current build status comes from the pinned [public EgoCom dataset release](https://huggingface.co/datasets/batgre/conversational-dynamics-egocom/tree/210a7decec4038ae8e2eb7c1013b2c1a20bedb49); it does **not** describe every local machine or the private Ego4D release.

## Verified evidence

- Registry: **164** labels in 18 families. EgoCom release materializes **87** (81 speech, 6 text, 0 social); **74** of these are audio-only by registry modality. Ego4D materialization is **not inspected** (private release inaccessible without authorization in this environment).
- The 17 public files (77,527,267 bytes, including action grid, indexes, and sidecars) were downloaded into the ignored `data/hf_snapshot/egocom/` directory and verified against the public file hashes. The three extractor manifests and release metadata agree on the action-grid SHA-256 `366b87c38d1e710d0c54a311011760e3b1e6b78226dc8c030da925170fb89d1a`; the speech/text Parquet tables match their manifest checksums.
- Public action grid has **1,387,790** rows. Index: **1,088,761 train**, **86,268 validation**, **209,436 test** rows. Audit values and validity rates are in the local ignored `reports/labels/egocom_hf_coverage.md` (from `coverage_report` over the fetched sidecars). The store is independent from raw media and does not establish causal action effects.

## Evaluation use

| Group | Current evidence and use |
| --- | --- |
| Vocal dynamics | Published EgoCom speech sidecar includes speaking state, overlap, onset context, floor changes, future 0.1/0.5/1.0 s targets and next speaker. Use future labels as *targets*, with the reference instant at the end of each 100 ms cell. Never feed them as past inputs. |
| Acoustic/prosodic controls | `prosody.ego_rms_db`, `prosody.ego_solo_mask_fraction` and eight `nuisance.*` audio controls are implemented but not published. Audio extractor requires canonical intermediate artifacts **and local raw media**. `prosody.ego_f0_hz` and `prosody.ego_voiced_fraction` additionally need `--external` and the `labels-audio` extra. None is independently validated as a conversational social label. |
| Transcripts | Six text sidecar labels are published for EgoCom. `text.speech_rate` is derived from word-level timing and annotated `text,audio`; token/punctuation labels carry text information even if observed through speech, so they do not qualify as audio-only input features. |
| Abstract audio concepts | Tone/mood, conversational ambiance and topic are **not** established audio labels in the current registry or in these sidecars. `social_states.affect_emotion` / `tension` and other social states require validated annotation and are unsupported; `metadata.background_conditions` has native **fan/music flags only**, not conversation mood or topic. Do not infer these concepts from timing or fan/music. |
| Audio nuisance | Mixed-signal RMS, zero-crossing, spectral and background-noise proxies are controls for confounding, not success targets for turn-taking. Native fan/music metadata is published for EgoCom. |
| Other unavailable concepts | Backchannel function, addressee, agreement and dominance need independently validated annotation/model. A short utterance or acoustic intensity is not a semantic label. Whole-recording `profiles.*` is diagnostic and includes future information, never a predictive input at an earlier point. |

## Rebuild prerequisites and commands

The public release contains derived labels and indexes but **no raw media or cleaned native annotations**. Therefore `conv-wm build labels --dataset all` cannot rebuild from that download. On this clean checkout it stops at missing `data/processed/vocal_action_grid/ego4d/vocal_action_grid.parquet`. The private Ego4D release returned HTTP 401 to an unauthenticated request. A complete source-faithful extraction requires each corpus's raw annotations, processed focal-voice/action-grid lineage, media manifest, and local raw media for the audio extractor.

```bash
# On a machine with the full corpus and upstream reports:
EGO_DATA_ROOT=/path/to/local/corpus-root uv run conv-wm build labels --dataset all
EGO_DATA_ROOT=/path/to/local/corpus-root uv run conv-wm build labels --dataset all --extractors audio
# Optionally add Praat pitch, still attributed only to ego-solo speech:
EGO_DATA_ROOT=/path/to/local/corpus-root uv sync --extra labels-audio
EGO_DATA_ROOT=/path/to/local/corpus-root uv run conv-wm build labels --dataset all --extractors audio --external
EGO_DATA_ROOT=/path/to/local/corpus-root uv run conv-wm audit labels --dataset all
EGO_DATA_ROOT=/path/to/local/corpus-root uv run conv-wm labels status --dataset all
```

Building an extractor is atomic and recorded by its `manifest.json`; audit status must be checked before claiming current materialization. The following table is a fixed snapshot of the pinned public EgoCom release. `Eligible` in the Ego4D column comes only from declared source facts; it is **not** a claim that Ego4D values have been extracted.

| Label | Modality | Source | EgoCom public | Ego4D facts |
| --- | --- | --- | --- | --- |
| `instantaneous.speaker_activity_subframes` | audio | native_annotation | published | eligible |
| `instantaneous.speaker_activity` | audio | native_annotation | published | eligible |
| `instantaneous.ego_speaking_subframes` | audio | native_annotation | published | eligible |
| `instantaneous.ego_speaking` | audio | native_annotation | published | eligible |
| `instantaneous.others_active_subframes` | audio | deterministic | published | eligible |
| `instantaneous.others_active` | audio | deterministic | published | eligible |
| `instantaneous.joint_speech_state_subframes` | audio | deterministic | published | eligible |
| `instantaneous.joint_speech_state_occupancy` | audio | deterministic | published | eligible |
| `instantaneous.active_speaker_count_subframes` | audio | deterministic | published | eligible |
| `instantaneous.floor_holder_subframes` | audio | deterministic | published | eligible |
| `instantaneous.diarized_speech_activity` | audio | external_model | unsupported | — |
| `events.speaker_onset_subframes` | audio | deterministic | published | eligible |
| `events.speaker_offset_subframes` | audio | deterministic | published | eligible |
| `events.speaker_transition_valid_subframes` | audio | deterministic | published | eligible |
| `events.ego_onset_subframes` | audio | deterministic | published | eligible |
| `events.ego_offset_subframes` | audio | deterministic | published | eligible |
| `events.other_onset_subframes` | audio | deterministic | published | eligible |
| `events.other_offset_subframes` | audio | deterministic | published | eligible |
| `events.floor_change_subframes` | audio | deterministic | published | eligible |
| `events.previous_floor_holder_subframes` | audio | deterministic | published | eligible |
| `events.next_floor_holder_subframes` | audio | deterministic | published | eligible |
| `events.speaker_onsets` | audio | deterministic | published | eligible |
| `events.speaker_offsets` | audio | deterministic | published | eligible |
| `onset_context.previous_unique_speaker_subframes` | audio | deterministic | published | eligible |
| `onset_context.ego_was_last_unique_speaker_subframes` | audio | deterministic | published | eligible |
| `onset_context.others_active_at_ego_onset_subframes` | audio | deterministic | published | eligible |
| `onset_context.silence_duration_before_ego_onset_subframes` | audio | deterministic | published | eligible |
| `onset_context.simultaneous_other_onset_subframes` | audio | deterministic | published | eligible |
| `onset_context.ego_onset_context_valid_subframes` | audio | deterministic | published | eligible |
| `onset_context.ego_onset_type_subframes` | audio | deterministic | published | eligible |
| `overlap.within_overlap_subframes` | audio | deterministic | published | eligible |
| `overlap.between_overlap_subframes` | audio | deterministic | published | eligible |
| `overlap.simultaneous_onset_overlap_subframes` | audio | deterministic | published | eligible |
| `overlap.overlap_events` | audio | deterministic | published | eligible |
| `overlap.overlap_function` | audio, text | human_annotation | unsupported | — |
| `timing.time_since_ego_onset` | audio | deterministic | published | eligible |
| `timing.time_since_ego_offset` | audio | deterministic | published | eligible |
| `timing.time_since_other_onset` | audio | deterministic | published | eligible |
| `timing.time_since_other_offset` | audio | deterministic | published | eligible |
| `timing.time_since_speaker_activity` | audio | deterministic | published | eligible |
| `timing.time_since_floor_change` | audio | deterministic | published | eligible |
| `timing.time_to_next_ego_onset` | audio | deterministic | published | eligible |
| `timing.time_to_next_ego_offset` | audio | deterministic | published | eligible |
| `timing.time_to_next_other_onset` | audio | deterministic | published | eligible |
| `timing.time_to_next_other_offset` | audio | deterministic | published | eligible |
| `timing.time_to_next_floor_change` | audio | deterministic | published | eligible |
| `timing.time_to_next_speaker_onset` | audio | deterministic | published | eligible |
| `timing.silence_duration` | audio | deterministic | published | eligible |
| `timing.observation_bounds` | audio | deterministic | published | eligible |
| `timing.silences` | audio | deterministic | published | eligible |
| `turns.speech_runs` | audio | deterministic | published | eligible |
| `turns.turns` | audio | deterministic | published | eligible |
| `turns.floor_transfers` | audio | deterministic | published | eligible |
| `next_speaker.next_speaker` | audio | deterministic | published | eligible |
| `next_speaker.next_unique_speaker` | audio | deterministic | published | eligible |
| `next_speaker.current_speaker_continues` | audio | deterministic | published | eligible |
| `next_speaker.floor_transfer_target` | audio | deterministic | published | eligible |
| `future.future_speaker_activity` | audio | deterministic | published | eligible |
| `future.future_ego_activity` | audio | deterministic | published | eligible |
| `future.future_others_activity` | audio | deterministic | published | eligible |
| `future.future_joint_speech_state` | audio | deterministic | published | eligible |
| `future.future_floor_holder` | audio | deterministic | published | eligible |
| `future.future_overlap` | audio | deterministic | published | eligible |
| `future.future_ego_onset` | audio | deterministic | published | eligible |
| `future.future_ego_offset` | audio | deterministic | published | eligible |
| `future.future_other_onset` | audio | deterministic | published | eligible |
| `future.future_other_offset` | audio | deterministic | published | eligible |
| `future.future_next_speaker` | audio | deterministic | published | eligible |
| `prosody.ego_f0_hz` | audio | external_model | not built | eligible |
| `prosody.ego_voiced_fraction` | audio | external_model | not built | eligible |
| `prosody.ego_rms_db` | audio | deterministic | not built | eligible |
| `prosody.ego_solo_mask_fraction` | audio | deterministic | not built | eligible |
| `prosody.others_f0` | audio | external_model | unsupported | — |
| `prosody.egemaps` | audio | external_model | unsupported | — |
| `prosody.mfcc` | audio | deterministic | unsupported | — |
| `prosody.voice_quality` | audio | external_model | unsupported | — |
| `prosody.final_lengthening` | audio, text | external_model | unsupported | — |
| `prosody.perceptual_loudness` | audio | deterministic | unsupported | — |
| `text.tokens` | text | native_annotation | published | eligible |
| `text.speech_rate` | text, audio | deterministic | published | — |
| `text.final_punctuation` | text | deterministic | published | eligible |
| `text.interrogative_cue` | text | deterministic | published | eligible |
| `text.turn_final_token` | text | deterministic | published | eligible |
| `text.discourse_markers` | text | deterministic | published | eligible |
| `text.lexical_completion` | text | external_model | unsupported | — |
| `text.syntactic_completion` | text | external_model | unsupported | — |
| `text.semantic_completion` | text | human_annotation | unsupported | — |
| `text.dialogue_act` | text | human_annotation | unsupported | — |
| `text.adjacency_pair_role` | text | human_annotation | unsupported | — |
| `text.agreement` | text | human_annotation | unsupported | — |
| `text.repair` | text | human_annotation | unsupported | — |
| `text.relevance` | text | human_annotation | unsupported | — |
| `text.common_ground` | text | human_annotation | unsupported | — |
| `text.asr_transcript` | audio, text | external_model | unsupported | — |
| `video.face_visible` | video | external_model | unsupported | — |
| `video.face_bbox` | video | external_model | unsupported | — |
| `video.face_landmarks` | video | external_model | unsupported | — |
| `video.head_pose` | video | external_model | unsupported | — |
| `video.gaze_proxy` | video | external_model | unsupported | — |
| `video.gaze_target` | video | external_model | unsupported | — |
| `video.mouth_openness` | video | external_model | unsupported | — |
| `video.nod_shake` | video | external_model | unsupported | — |
| `video.facial_movement` | video | external_model | unsupported | — |
| `video.upper_body_pose` | video | external_model | unsupported | — |
| `video.body_orientation` | video | external_model | unsupported | — |
| `video.hand_pose` | video | external_model | unsupported | — |
| `video.gesture_activity` | video | external_model | unsupported | — |
| `video.pre_speech_movement` | video | external_model | unsupported | — |
| `video.participant_geometry` | video | external_model | unsupported | — |
| `video.camera_motion` | video | external_model | unsupported | — |
| `social_native.looking_at_wearer_subframes` | video | native_annotation | not offered | eligible |
| `social_native.talking_to_wearer_subframes` | audio, video | native_annotation | not offered | eligible |
| `social_native.anyone_looking_at_wearer_subframes` | video | deterministic | not offered | eligible |
| `social_native.anyone_talking_to_wearer_subframes` | audio, video | deterministic | not offered | eligible |
| `social_native.face_tracked_subframes` | video | native_annotation | not offered | eligible |
| `social_native.face_track_bbox` | video | native_annotation | not offered | eligible |
| `social_native.social_segments` | audio, video | native_annotation | not offered | eligible |
| `social_native.face_tracks` | video | native_annotation | not offered | eligible |
| `addressee.addressee` | audio, video, text | human_annotation | unsupported | — |
| `addressee.broadcast` | audio, video, text | human_annotation | unsupported | — |
| `backchannel.events` | audio, video, text | human_annotation | unsupported | — |
| `profiles.speaking_time` | audio | deterministic | published | eligible |
| `profiles.turn_statistics` | audio | deterministic | published | eligible |
| `profiles.pause_statistics` | audio | deterministic | published | eligible |
| `profiles.onset_offset_counts` | audio | deterministic | published | eligible |
| `profiles.overlap_statistics` | audio | deterministic | published | eligible |
| `profiles.floor_transfer_statistics` | audio | deterministic | published | eligible |
| `profiles.interaction_profile_features` | audio | deterministic | published | eligible |
| `profiles.interaction_profile_feature_names` | audio | deterministic | published | eligible |
| `social_states.engagement` | audio, video, text | human_annotation | unsupported | — |
| `social_states.dominance` | audio, video, text | human_annotation | unsupported | — |
| `social_states.leadership` | audio, video, text | human_annotation | unsupported | — |
| `social_states.rapport` | audio, video, text | human_annotation | unsupported | — |
| `social_states.cohesion` | audio, video, text | human_annotation | unsupported | — |
| `social_states.tension` | audio, video, text | human_annotation | unsupported | — |
| `social_states.awkwardness` | audio, video, text | human_annotation | unsupported | — |
| `social_states.stance` | audio, video, text | human_annotation | unsupported | — |
| `social_states.affect_emotion` | audio, video, text | human_annotation | unsupported | — |
| `social_states.agreement_conflict` | audio, video, text | human_annotation | unsupported | — |
| `social_states.floor_partition` | audio, video, text | human_annotation | unsupported | — |
| `nuisance.global_audio_rms` | audio | deterministic | not built | eligible |
| `nuisance.zero_crossing_rate` | audio | deterministic | not built | eligible |
| `nuisance.spectral_centroid` | audio | deterministic | not built | eligible |
| `nuisance.spectral_bandwidth` | audio | deterministic | not built | eligible |
| `nuisance.spectral_flux` | audio | deterministic | not built | eligible |
| `nuisance.spectral_valid` | audio | deterministic | not built | eligible |
| `nuisance.background_noise_proxy` | audio | deterministic | not built | eligible |
| `nuisance.audio_valid` | audio | deterministic | not built | eligible |
| `nuisance.frame_mean_rgb` | video | deterministic | not built | eligible |
| `nuisance.brightness` | video | deterministic | not built | eligible |
| `nuisance.contrast` | video | deterministic | not built | eligible |
| `nuisance.saturation` | video | deterministic | not built | eligible |
| `nuisance.blur_score` | video | deterministic | not built | eligible |
| `nuisance.dominant_colour` | video | deterministic | not built | eligible |
| `nuisance.frame_difference` | video | deterministic | not built | eligible |
| `nuisance.frame_valid` | video | deterministic | not built | eligible |
| `metadata.recording_identity` | metadata | native_annotation | published | eligible |
| `metadata.participants` | metadata | native_annotation | published | eligible |
| `metadata.recording_span` | metadata | native_annotation | published | eligible |
| `metadata.source_offset` | metadata | native_annotation | not offered | eligible |
| `metadata.background_conditions` | metadata | native_annotation | published | — |
| `metadata.participant_identity` | metadata | native_annotation | published | eligible |
| `metadata.participant_native_speaker` | metadata | native_annotation | published | — |
| `metadata.participant_is_host` | metadata | native_annotation | published | — |
