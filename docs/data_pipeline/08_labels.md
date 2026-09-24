# Label and cue ontology

The v0 vocal action labels are implemented: see
[`native_focal_voice_state.md`](native_focal_voice_state.md) for the wearer's
continuous `SPEAKING` / `SILENT` / `UNKNOWN` state,
[`control_focal_voice_state.md`](control_focal_voice_state.md) for the same
state requantized at Δ = 100 ms, and
[`vocal_action_grid.md`](vocal_action_grid.md) for the Δ = 100 ms
`NO_EVENT` / `ONSET` / `OFFSET` action grid derived from it.

Every other label — multi-participant speech activity, floor, events, overlap,
timing, turns, next speaker, future targets, native social annotations
(Looking-At-Me, Talking-To-Me), transcript cues, prosody and nuisance
controls — lives in optional, versioned **label sidecars** next to the action
grid: [`labels.md`](labels.md) explains how they are built and requested, and
[`labels_registry.md`](labels_registry.md) (generated from the registry)
describes every label, including the interpretations that are registered but
deliberately not materialized.
