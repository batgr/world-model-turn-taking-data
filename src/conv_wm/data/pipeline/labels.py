"""Build the label sidecars: canonical facts + action grid -> versioned label artifacts.

```text
cleaned annotations ─> adapter.label_facts ─> canonical facts ─┐
vocal action grid (checked) ───────────────────────────────────┼─> speech / social / text
media manifest (checked, media extractors only) ───────────────┘   audio / video
```

One extractor is the unit of building: each writes one directory of tables
and a deterministic ``manifest.json`` (:mod:`conv_wm.data.labels.store`) and
one provenance report below ``${paths.reports}/labels/<dataset>/<extractor>``.
The default build runs the annotation-only extractors; the media ones decode
audio or video and run only when requested; ``external_model`` labels run only
with ``external=True``.

Guarantees:

* the grid rows are exactly the action grid's, which is read through its
  report (a stale grid is refused);
* the interim annotation tables the action grid was built from must still be
  byte-identical (a stale upstream is refused);
* the wearer's speech derived here must equal the native focal voice state
  the grid descends from (checked per recording, a mismatch fails the build);
* the same inputs, configuration and code produce byte-identical tables and
  manifests.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
from omegaconf import DictConfig, OmegaConf

from conv_wm.config import PROJECT_ROOT, pipeline_paths
from conv_wm.data import datasets
from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.audits.media_metadata import MEDIA_METADATA_TABLE
from conv_wm.data.labels import facts as label_facts
from conv_wm.data.labels import store
from conv_wm.data.labels.catalog import REGISTRY, extractor_labels
from conv_wm.data.labels.facts import LabelSource
from conv_wm.data.labels.gridding import GridFrame
from conv_wm.data.labels.registry import (
    LABEL_SCHEMA_VERSION,
    REGISTRY_VERSION,
    Extractor,
    SourceKind,
    Table,
)
from conv_wm.data.labels.selection import LabelSelection, resolve
from conv_wm.data.labels.social import (
    SOCIAL_SEGMENT_SCHEMA,
    social_grid,
    social_segment_rows,
)
from conv_wm.data.labels.speech import (
    GridKeys,
    derive_structures,
    grid_keys_by_recording,
    key_columns,
    rows_table,
    speech_tables,
)
from conv_wm.data.labels.text import (
    TEXT_RULES_VERSION,
    TEXT_UNIT_SCHEMA,
    TOKEN_EVENT_SCHEMA,
    text_rows,
)
from conv_wm.data.labels.timeline import (
    LABEL_RULES_VERSION,
    SPEAKING,
    UNKNOWN,
    LabelConfig,
    RecordingStructure,
)
from conv_wm.data.pipeline.control_state import config_checksum
from conv_wm.data.pipeline.model_ready import load_action_grid_input
from conv_wm.data.pipeline_inputs import (
    CheckedArtifact,
    artifact_reference,
    sha256_file,
)
from conv_wm.data.vocal.action_grid import DECISION_STEP_S
from conv_wm.provenance import collect_provenance
from conv_wm.reports import JsonDict, write_summary

logger = logging.getLogger(__name__)

REPORT_SCHEMA_VERSION = 1
PROCESSED_ROOT = Path("labels")
REPORT_ROOT = Path("labels")
REPORT_FILE = "report.json"
BUILD_COMMAND = "conv-wm build labels"
DEFAULT_EXTRACTORS = (Extractor.SPEECH, Extractor.SOCIAL, Extractor.TEXT)
"""Annotation-only extractors: what ``conv-wm build labels`` runs by default."""
LINEAGE_CHAIN = (
    "native annotation",
    "native focal voice state",
    "control focal voice state",
    "vocal action grid",
    "label sidecars",
)
EXTRACTOR_VERSIONS: Mapping[Extractor, str] = {
    Extractor.SPEECH: f"speech-rules-v{LABEL_RULES_VERSION}",
    Extractor.SOCIAL: f"social-rules-v{LABEL_RULES_VERSION}",
    Extractor.TEXT: f"text-rules-v{TEXT_RULES_VERSION}",
    Extractor.AUDIO: "audio-features-v1",
    Extractor.VIDEO: "video-features-v1",
}
"""Bumped (in the owning module) when an extractor's derivation changes."""
NATIVE_AGREEMENT_TOLERANCE_S = 1e-6


class StaleUpstreamError(ValueError):
    """An upstream artifact changed after the action grid was built from it."""


def label_config(cfg: DictConfig) -> LabelConfig:
    """The ``labels`` section of the configuration (defaults when absent)."""
    section = cfg.get("labels")
    if section is None:
        return LabelConfig()
    values = OmegaConf.to_container(section, resolve=True)
    assert isinstance(values, dict)
    return LabelConfig.from_mapping(values)


def config_digest(config: LabelConfig) -> str:
    """SHA-256 of the canonical label configuration: what invalidates a build."""
    return hashlib.sha256(
        store.canonical_json(config.to_dict()).encode("utf-8")
    ).hexdigest()


def supported_datasets() -> tuple[str, ...]:
    """Registered datasets that declare label facts."""
    return tuple(spec.name for spec in datasets.iter_specs() if spec.label_facts)


def selected_datasets(dataset: str) -> tuple[str, ...]:
    """Expand the CLI selector (``all`` or one supported dataset)."""
    supported = supported_datasets()
    if dataset == "all":
        return supported
    if dataset not in supported:
        raise ValueError(
            f"dataset must be one of {[*supported, 'all']}, got {dataset!r}"
        )
    return (dataset,)


def dataset_dir(cfg: DictConfig, dataset: str) -> Path:
    """Where a dataset's label store lives."""
    return pipeline_paths(cfg).processed / PROCESSED_ROOT / dataset


def extractors_for(
    *,
    extractors: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    modalities: Sequence[str] | None = None,
) -> tuple[Extractor, ...]:
    """Which extractors a build request needs.

    Explicit ``extractors`` win. Otherwise ``labels`` / ``modalities`` select
    labels with the consumer's selection rules and every extractor holding one
    of them is built — media extractors included. With neither, the
    annotation-only :data:`DEFAULT_EXTRACTORS` are built.
    """
    if extractors:
        return tuple(Extractor(name) for name in dict.fromkeys(extractors))
    if not labels and not modalities:
        return DEFAULT_EXTRACTORS
    selection = LabelSelection(True, tuple(labels or ("all",)), tuple(modalities or ()))
    chosen = {spec.extractor for spec in resolve(selection, REGISTRY).labels}
    return tuple(extractor for extractor in Extractor if extractor in chosen)


@dataclass(frozen=True)
class ExtractorOutput:
    """What one extractor build wrote for one dataset."""

    dataset: str
    extractor: Extractor
    manifest: JsonDict
    report: JsonDict
    directory: Path
    report_path: Path


def _require(path: Path, command: str) -> Path:
    if not path.exists():
        raise MissingPrerequisiteError(path, produce_with=command)
    return path


def check_upstream_annotations(grid: CheckedArtifact) -> list[dict[str, str]]:
    """Refuse interim annotation tables that changed after the grid was built."""
    recorded = (grid.report.get("source_annotation_versions") or {}).get(
        "native_annotations"
    ) or []
    for entry in recorded:
        path = Path(entry["path"])
        if not path.exists():
            raise MissingPrerequisiteError(
                path, produce_with="conv-wm clean annotations"
            )
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise StaleUpstreamError(
                f"{path} changed after the action grid was built (checksum {digest}, "
                f"recorded {entry['sha256']}); rerun conv-wm build all --dataset "
                f"{grid.dataset}"
            )
    return list(recorded)


def native_agreement(
    structures: Mapping[str, RecordingStructure], native: pd.DataFrame
) -> dict[str, Any]:
    """Compare the wearer's derived speech with the native focal voice state.

    Both must describe the same time as SPEAKING and the same time as
    UNKNOWN in every recording; any disagreement means the label facts and
    the grid were derived from different evidence.
    """
    mismatched: list[str] = []
    grouped = {str(k): g for k, g in native.groupby("recording_id", sort=False)}
    for recording_id, structure in structures.items():
        rows = grouped.get(recording_id)
        if rows is None:
            mismatched.append(recording_id)
            continue
        durations = rows["canonical_end_s"] - rows["canonical_start_s"]
        native_speaking = float(durations[rows["voice_state"].eq("SPEAKING")].sum())
        native_unknown = float(durations[rows["voice_state"].eq("UNKNOWN")].sum())
        pieces = structure.pieces
        lengths = pieces.ends - pieces.starts
        ego = pieces.states[:, 0]
        if (
            abs(float(lengths[ego == SPEAKING].sum()) - native_speaking)
            > NATIVE_AGREEMENT_TOLERANCE_S
            or abs(float(lengths[ego == UNKNOWN].sum()) - native_unknown)
            > NATIVE_AGREEMENT_TOLERANCE_S
        ):
            mismatched.append(recording_id)
    return {
        "recordings_compared": len(structures),
        "recordings_mismatched": len(mismatched),
        "examples": sorted(mismatched)[:10],
        "tolerance_s": NATIVE_AGREEMENT_TOLERANCE_S,
    }


def load_native_timeline(grid: CheckedArtifact) -> pd.DataFrame | None:
    """The native focal voice state the grid descends from, checksum-verified."""
    reference = (grid.report.get("input_artifacts") or {}).get(
        "native_focal_voice_state_timeline"
    )
    if not reference:
        return None
    path = Path(reference["path"])
    if not path.exists():
        return None
    if sha256_file(path) != reference["sha256"]:
        raise StaleUpstreamError(
            f"{path} changed after the action grid was built; rerun conv-wm build all "
            f"--dataset {grid.dataset}"
        )
    return pd.read_parquet(path)


def _availability(
    extractor: Extractor, provides: frozenset[str], external: bool
) -> tuple[list[str], dict[str, str]]:
    materialized: list[str] = []
    unavailable: dict[str, str] = {}
    for spec in extractor_labels(extractor):
        missing = sorted(set(spec.requires) - provides)
        if missing:
            unavailable[spec.name] = f"the dataset does not provide the facts {missing}"
        elif spec.source_kind is SourceKind.EXTERNAL_MODEL and not external:
            unavailable[spec.name] = (
                f"external_model label: rebuild with conv-wm build labels "
                f"--extractors {extractor} --external"
            )
        else:
            materialized.append(spec.name)
    return materialized, unavailable


def _inputs(
    grid: CheckedArtifact, source: LabelSource, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Portable input lineage for a manifest: file names and checksums only."""
    return {
        "action_grid": {"file": grid.table_path.name, "sha256": grid.table_sha256},
        "action_grid_report": {
            "file": grid.report_path.name,
            "sha256": grid.report_sha256,
        },
        "annotations": [
            {"file": path.name, "sha256": sha256_file(path)}
            for path in source.annotation_paths
        ],
        **(extra or {}),
    }


def _manifest(
    *,
    dataset: str,
    extractor: Extractor,
    config: LabelConfig,
    materialized: list[str],
    unavailable: dict[str, str],
    inputs: dict[str, Any],
    statistics: Mapping[str, Any],
    source: LabelSource,
    external_tools: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    commit, dirty = _code_revision()
    return {
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "registry_version": REGISTRY_VERSION,
        "dataset": dataset,
        "extractor": str(extractor),
        "extractor_version": EXTRACTOR_VERSIONS[extractor],
        "config": config.to_dict(),
        "config_digest": config_digest(config),
        "time": {
            "clock": "the recording's canonical clock (native focal voice state)",
            "decision_step_s": DECISION_STEP_S,
            "subframes_per_step": config.subframes_per_step,
            "subframe_s": DECISION_STEP_S / config.subframes_per_step,
            "cell": "[t_k, t_k + step), t_k = decision_index * step",
            "reference_instant": "cell end r_k = t_k + step (timing, next speaker, future)",
            "future_horizons_s": list(config.future_horizons_s),
            "events": "native timestamps, never rounded",
        },
        "materialized_labels": materialized,
        "unavailable_labels": dict(sorted(unavailable.items())),
        "inputs": inputs,
        "source": {
            "annotation_schema_version": source.annotation_schema_version,
            "cleaning_rule_version": source.cleaning_rule_version,
            "provides": sorted(source_provides(dataset)),
            "limitations": list(source.limitations),
        },
        "external_tools": dict(external_tools or {}),
        "statistics": dict(statistics),
        "code_revision": {"git_commit": commit, "git_dirty": dirty},
    }


def _code_revision() -> tuple[str | None, bool | None]:
    from conv_wm.provenance import git_state

    return git_state()


def source_provides(dataset: str) -> frozenset[str]:
    """Facts a registered dataset declares for labels."""
    spec = datasets.get(dataset).label_facts
    return spec.provides if spec else frozenset()


def _social_tables(
    structures: Mapping[str, RecordingStructure],
    grid_keys: Mapping[str, GridKeys],
    source: LabelSource,
    config: LabelConfig,
) -> dict[Table, pa.Table]:
    social = (
        source.social
        if source.social is not None
        else _empty(label_facts.SOCIAL_COLUMNS)
    )
    tracks = (
        source.tracks
        if source.tracks is not None
        else _empty(label_facts.TRACK_COLUMNS)
    )
    label_facts.validate_frame(social, label_facts.SOCIAL_COLUMNS, "social")
    label_facts.validate_frame(tracks, label_facts.TRACK_COLUMNS, "track")
    social_by = {str(k): g for k, g in social.groupby("recording_id", sort=False)}
    tracks_by = {str(k): g for k, g in tracks.groupby("recording_id", sort=False)}
    grids: list[pa.Table] = []
    segments: list[dict[str, Any]] = []
    for recording_id in sorted(structures):
        structure = structures[recording_id]
        keys = grid_keys[recording_id]
        frame = GridFrame(
            keys.decision_index, DECISION_STEP_S, config.subframes_per_step
        )
        own_social = social_by.get(recording_id, social.iloc[0:0])
        own_tracks = tracks_by.get(recording_id, tracks.iloc[0:0])
        grids.append(
            pa.table(
                {
                    **key_columns(recording_id, keys),
                    **social_grid(structure, frame, own_social, own_tracks),
                }
            )
        )
        segments.extend(
            social_segment_rows(
                structure, own_social, own_tracks, structure.recording.frame_rate_hz
            )
        )
    segments.sort(key=lambda r: (r["recording_id"], r["segment_type"], r["segment_id"]))
    return {
        Table.GRID: pa.concat_tables(grids),
        Table.SEGMENTS: rows_table(segments, SOCIAL_SEGMENT_SCHEMA),
    }


def _text_tables(
    structures: Mapping[str, RecordingStructure], source: LabelSource
) -> dict[Table, pa.Table]:
    tokens = (
        source.tokens
        if source.tokens is not None
        else _empty(label_facts.TOKEN_COLUMNS)
    )
    label_facts.validate_frame(tokens, label_facts.TOKEN_COLUMNS, "token")
    by_recording = {str(k): g for k, g in tokens.groupby("recording_id", sort=False)}
    events: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for recording_id in sorted(structures):
        own = by_recording.get(recording_id, tokens.iloc[0:0])
        token_events, text_units = text_rows(structures[recording_id], own)
        events.extend(token_events)
        units.extend(text_units)
    return {
        Table.EVENTS: rows_table(events, TOKEN_EVENT_SCHEMA),
        Table.SEGMENTS: rows_table(units, TEXT_UNIT_SCHEMA),
    }


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})


@dataclass
class _DatasetInputs:
    """Everything one dataset's extractors share, loaded once."""

    grid: CheckedArtifact
    source: LabelSource
    grid_keys: dict[str, GridKeys]
    structures: dict[str, RecordingStructure]
    annotation_lineage: list[dict[str, str]]
    agreement: dict[str, Any]


def _load_inputs(cfg: DictConfig, dataset: str, config: LabelConfig) -> _DatasetInputs:
    paths = pipeline_paths(cfg)
    grid = load_action_grid_input(cfg, dataset)
    lineage = check_upstream_annotations(grid)
    media_path = _require(paths.reports / MEDIA_METADATA_TABLE, "conv-wm audit media")
    spec = datasets.get(dataset).label_facts
    assert spec is not None
    source = spec.load(cfg, pd.read_parquet(media_path))
    grid_keys = grid_keys_by_recording(
        grid.table[["recording_id", "decision_index", "decision_time_s"]]
    )
    structures = derive_structures(source.recordings, grid_keys, config)
    native = load_native_timeline(grid)
    agreement = (
        native_agreement(structures, native)
        if native is not None
        else {"recordings_compared": 0, "note": "native timeline unavailable"}
    )
    if agreement.get("recordings_mismatched"):
        raise StaleUpstreamError(
            f"{dataset}: the wearer's speech in the label facts differs from the native "
            f"focal voice state in {agreement['recordings_mismatched']} recordings "
            f"(e.g. {agreement['examples']}); rerun conv-wm build all --dataset {dataset}"
        )
    return _DatasetInputs(grid, source, grid_keys, structures, lineage, agreement)


def build_extractor(
    cfg: DictConfig,
    dataset: str,
    extractor: Extractor,
    inputs: _DatasetInputs,
    config: LabelConfig,
    *,
    external: bool,
    command: str,
) -> ExtractorOutput:
    """Build one extractor of one dataset and write its directory and report."""
    provides = source_provides(dataset)
    materialized, unavailable = _availability(extractor, provides, external)
    tables: dict[Table, pa.Table] = {}
    statistics: dict[str, Any] = {}
    extra_inputs: dict[str, Any] = {}
    tools: dict[str, Any] = {}
    if materialized:
        if extractor is Extractor.SPEECH:
            result = speech_tables(
                inputs.structures, inputs.grid_keys, config, step_s=DECISION_STEP_S
            )
            tables, statistics = result.tables, result.statistics
        elif extractor is Extractor.SOCIAL:
            tables = _social_tables(
                inputs.structures, inputs.grid_keys, inputs.source, config
            )
        elif extractor is Extractor.TEXT:
            tables = _text_tables(inputs.structures, inputs.source)
        else:
            raise NotImplementedError(
                f"the {extractor} extractor is not implemented yet"
            )
        statistics = {
            **statistics,
            **{f"{table}_rows": content.num_rows for table, content in tables.items()},
        }
    directory = dataset_dir(cfg, dataset) / str(extractor)
    manifest = store.write_extractor(
        directory,
        tables,
        _manifest(
            dataset=dataset,
            extractor=extractor,
            config=config,
            materialized=materialized,
            unavailable=unavailable,
            inputs=_inputs(inputs.grid, inputs.source, extra_inputs),
            statistics=statistics,
            source=inputs.source,
            external_tools=tools,
        ),
    )
    store.write_registry(dataset_dir(cfg, dataset))
    report_path = (
        pipeline_paths(cfg).reports
        / REPORT_ROOT
        / dataset
        / str(extractor)
        / REPORT_FILE
    )
    provenance = collect_provenance()
    uv_lock = PROJECT_ROOT / "uv.lock"
    report = write_summary(
        report_path,
        {
            "build_type": "labels",
            "schema_version": REPORT_SCHEMA_VERSION,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "registry_version": REGISTRY_VERSION,
            "dataset": dataset,
            "extractor": str(extractor),
            "created_at": provenance.generated_at_utc,
            "git_commit": provenance.git_commit,
            "git_dirty": provenance.git_dirty,
            "command": command,
            "config_checksum": config_checksum(cfg),
            "label_config_digest": manifest["config_digest"],
            "lineage_chain": list(LINEAGE_CHAIN),
            "input_artifacts": {
                "vocal_action_grid": artifact_reference(inputs.grid.table_path),
                "vocal_action_grid_report": artifact_reference(inputs.grid.report_path),
                "annotations": [
                    artifact_reference(p) for p in inputs.source.annotation_paths
                ],
                "native_annotations_verified": inputs.annotation_lineage,
            },
            "uv_lock_checksum": sha256_file(uv_lock) if uv_lock.exists() else None,
            "output_artifacts": {
                "manifest": artifact_reference(directory / store.MANIFEST_FILE),
                **{
                    name: {
                        "path": str(directory / entry["file"]),
                        "sha256": entry["sha256"],
                    }
                    for name, entry in manifest["tables"].items()
                },
            },
            "native_focal_voice_state_agreement": inputs.agreement,
            "materialized_label_count": len(materialized),
            "unavailable_label_count": len(unavailable),
            "statistics": statistics,
            "adapter_statistics": inputs.source.statistics,
            "limitations": list(inputs.source.limitations),
        },
        parameters=manifest["config"],
        provenance=provenance,
    )
    return ExtractorOutput(dataset, extractor, manifest, report, directory, report_path)


def run_label_build(
    cfg: DictConfig,
    *,
    dataset: str = "all",
    extractors: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    modalities: Sequence[str] | None = None,
    external: bool = False,
    command: str | None = None,
) -> list[ExtractorOutput]:
    """Build the requested extractors of every selected dataset."""
    config = label_config(cfg)
    chosen = extractors_for(extractors=extractors, labels=labels, modalities=modalities)
    invoked = command or f"{BUILD_COMMAND} --dataset {dataset}"
    outputs: list[ExtractorOutput] = []
    for name in selected_datasets(dataset):
        inputs = _load_inputs(cfg, name, config)
        logger.info(
            "%s: %d recordings, extractors %s",
            name,
            len(inputs.structures),
            [str(e) for e in chosen],
        )
        for extractor in chosen:
            outputs.append(
                build_extractor(
                    cfg,
                    name,
                    extractor,
                    inputs,
                    config,
                    external=external,
                    command=invoked,
                )
            )
    return outputs


def label_status(cfg: DictConfig, dataset: str) -> dict[str, dict[str, Any]]:
    """Whether each built extractor of ``dataset`` is still current.

    ``stale`` reasons: the label configuration changed, the action grid was
    rebuilt, the registry or label schema changed, or a table no longer
    matches its manifest.
    """
    root = dataset_dir(cfg, dataset)
    config = config_digest(label_config(cfg))
    grid_report = (
        pipeline_paths(cfg).reports / "vocal_action_grid" / dataset / "report.json"
    )
    grid_sha = None
    if grid_report.exists():
        recorded = json.loads(grid_report.read_text(encoding="utf-8"))
        grid_sha = recorded.get("output_artifacts", {}).get("grid", {}).get("sha256")
    status: dict[str, dict[str, Any]] = {}
    for extractor in Extractor:
        manifest = store.read_manifest(root, extractor)
        if manifest is None:
            continue
        reasons: list[str] = []
        if manifest.get("config_digest") != config:
            reasons.append("label configuration changed")
        if manifest.get("inputs", {}).get("action_grid", {}).get("sha256") != grid_sha:
            reasons.append("action grid changed")
        if manifest.get("registry_version") != REGISTRY_VERSION:
            reasons.append("registry version changed")
        if manifest.get("label_schema_version") != LABEL_SCHEMA_VERSION:
            reasons.append("label schema changed")
        try:
            store.verify_extractor(root, manifest)
        except store.StaleLabelsError as error:
            reasons.append(str(error))
        status[str(extractor)] = {"current": not reasons, "reasons": reasons}
    return status


def release_label_store(
    source: Path, destination: Path, *, grid_sha256: str
) -> JsonDict:
    """Copy a dataset's built label store into a release, refusing a stale one.

    Every extractor must still match its manifest and must have been built
    from ``grid_sha256`` — the action grid the release ships. Returns the
    dataset-card section describing what was copied (no local path).
    """
    import shutil

    extractors: dict[str, Any] = {}
    manifests = [
        manifest
        for extractor in Extractor
        if (manifest := store.read_manifest(source, extractor)) is not None
    ]
    for manifest in manifests:
        store.verify_extractor(source, manifest)
        recorded = manifest.get("inputs", {}).get("action_grid", {}).get("sha256")
        if recorded != grid_sha256:
            raise StaleUpstreamError(
                f"{source / manifest['extractor']} was built from another action grid; "
                f"rerun {BUILD_COMMAND} --dataset {manifest['dataset']}"
            )
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copyfile(source / store.REGISTRY_FILE, destination / store.REGISTRY_FILE)
    for manifest in manifests:
        name = manifest["extractor"]
        shutil.copytree(source / name, destination / name)
        extractors[name] = {
            "extractor_version": manifest["extractor_version"],
            "config_digest": manifest["config_digest"],
            "materialized_labels": len(manifest["materialized_labels"]),
            "tables": {
                table: {"file": f"{name}/{entry['file']}", "sha256": entry["sha256"]}
                for table, entry in manifest["tables"].items()
            },
            "external_tools": sorted(manifest.get("external_tools", {})),
        }
    return {
        "registry": store.REGISTRY_FILE,
        "registry_version": REGISTRY_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "extractors": extractors,
        "raw_media_distributed": False,
    }


COVERAGE_DIR = REPORT_ROOT / "coverage"


@dataclass(frozen=True)
class CoverageOutputs:
    """The machine- and human-readable coverage report."""

    report: JsonDict
    json_path: Path
    markdown_path: Path


def run_label_coverage_audit(
    cfg: DictConfig, *, dataset: str = "all"
) -> CoverageOutputs:
    """Measure coverage, validity and sanity statistics of the built label stores."""
    from conv_wm.data.labels.coverage import coverage_markdown, coverage_report

    names = selected_datasets(dataset)
    stores = {name: dataset_dir(cfg, name) for name in names}
    missing = [name for name, path in stores.items() if not path.exists()]
    if len(missing) == len(stores):
        raise MissingPrerequisiteError(
            stores[missing[0]], produce_with=f"{BUILD_COMMAND} --dataset {dataset}"
        )
    report = coverage_report(
        {name: path for name, path in stores.items() if path.exists()},
        {name: source_provides(name) for name in names},
    )
    report["status"] = {name: label_status(cfg, name) for name in names}
    directory = pipeline_paths(cfg).reports / COVERAGE_DIR
    json_path = directory / (
        "coverage.json" if dataset == "all" else f"coverage_{dataset}.json"
    )
    written = write_summary(json_path, report)
    markdown_path = json_path.with_suffix(".md")
    markdown_path.write_text(coverage_markdown(report), encoding="utf-8")
    return CoverageOutputs(written, json_path, markdown_path)


def format_outputs(outputs: list[ExtractorOutput]) -> str:
    """One line per (dataset, extractor) with its label and row counts."""
    lines = []
    for output in outputs:
        tables = output.manifest.get("tables", {})
        rows = ", ".join(
            f"{name} {entry['rows']}" for name, entry in sorted(tables.items())
        )
        lines.append(
            f"{output.dataset}/{output.extractor}: "
            f"{len(output.manifest['materialized_labels'])} labels "
            f"({len(output.manifest['unavailable_labels'])} unavailable)"
            + (f"; {rows}" if rows else "")
            + f" -> {output.directory}"
        )
    return "\n".join(lines)


__all__ = [
    "BUILD_COMMAND",
    "DEFAULT_EXTRACTORS",
    "CoverageOutputs",
    "ExtractorOutput",
    "StaleUpstreamError",
    "build_extractor",
    "check_upstream_annotations",
    "config_digest",
    "dataset_dir",
    "extractors_for",
    "format_outputs",
    "label_config",
    "label_status",
    "native_agreement",
    "release_label_store",
    "run_label_build",
    "run_label_coverage_audit",
    "selected_datasets",
    "supported_datasets",
]
