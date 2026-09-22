# Label and cue ontology

The v0 vocal labels are implemented. See
[`native_focal_voice_state.md`](native_focal_voice_state.md) for the wearer's
continuous `SPEAKING` / `SILENT` / `UNKNOWN` state,
[`control_focal_voice_state.md`](control_focal_voice_state.md) for the same
state requantized at Δ = 100 ms, and
[`vocal_action_grid.md`](vocal_action_grid.md) for the Δ = 100 ms
`NO_EVENT` / `ONSET` / `OFFSET` action grid derived from it.

Non-vocal cues (gaze, addressee, visual attention) remain a planned capability.
Annotation sources carry their provenance and semantics (see
`05_annotation_audit.md`) so those labels can be derived later with explicit
provenance.
