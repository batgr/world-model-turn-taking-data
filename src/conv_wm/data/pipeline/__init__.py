"""The canonical pipeline stages, in the order they run.

```text
clean           raw annotations   -> validated interim tables
native_state    interim tables    -> SPEAKING / SILENT / UNKNOWN per recording
control_state   native timeline   -> the same, requantized at the control step
action_grid     control timeline  -> NO_EVENT / ONSET / OFFSET per 100 ms slot
model_ready     action grid       -> training anchors, splits, dataset card
```

Each module orchestrates one stage: it reads the previous artifact, refuses it
if its checksum no longer matches its report, writes tables and a lineage
report. The semantics they apply live in :mod:`conv_wm.data.vocal`; the
dataset-specific parts live in :mod:`conv_wm.data.datasets`.
"""
