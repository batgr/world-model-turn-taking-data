"""Wearer vocal state.

v0 pipeline: ``native_state`` (annotation -> SPEAKING/SILENT/UNKNOWN timeline),
``native_source`` (what dataset adapters hand to the build), ``control_state``
(native timeline -> controller-representable timeline at Δ) and ``action_grid``
(control timeline -> NO_EVENT/ONSET/OFFSET slots); ``intervals`` holds the
interval helpers the labels share.
"""
