"""Wearer vocal state.

v0 pipeline: ``native_state`` (annotation -> SPEAKING/SILENT/UNKNOWN timeline),
``native_source`` (what dataset adapters hand to the build), ``control_state``
(native timeline -> controller-representable timeline at Δ) and ``action_grid``
(control timeline -> NO_EVENT/ONSET/OFFSET slots). Every other
module here — ``audio``, ``detector``, ``identity``, ``coverage``, ``records``,
``sources``, ``config``, ``intervals`` — serves the diagnostic vocal annotation
coverage audit only.
"""
