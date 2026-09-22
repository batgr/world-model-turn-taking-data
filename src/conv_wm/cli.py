"""``conv-wm``: the single command-line entry point of the data pipeline.

Every subcommand calls the same audit functions that the tests exercise; this
module only parses arguments, resolves configuration and reports outcomes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from conv_wm.config import DEFAULT_CONFIG_PATH, load_config
from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.media.ffprobe import FFprobeError

EXIT_OK = 0
EXIT_FAILURE = 1
"""The audit could not run (missing prerequisite, missing tool, crash)."""
EXIT_AUDIT_FAILED = 2
"""The audit ran and its verdict is FAIL (structural or annotation contracts)."""


@dataclass(frozen=True)
class AuditCommand:
    """One ``conv-wm audit <name>`` subcommand."""

    name: str
    help: str
    run: Callable[[DictConfig, argparse.Namespace], int]
    supports_workers: bool = False
    supports_dataset: bool = False
    supports_coverage_config: bool = False


def _audit_manifest(cfg: DictConfig, _: argparse.Namespace) -> int:
    from conv_wm.data.audits.manifest import run_manifest_audit

    outputs = run_manifest_audit(cfg)
    print(f"manifest: {outputs.manifest_path} ({outputs.summary['n_files']} files)")
    print(f"summary: {outputs.summary_path}")
    return EXIT_OK


def _audit_structure(cfg: DictConfig, _: argparse.Namespace) -> int:
    from conv_wm.data.audits.structural import (
        format_structural_summary,
        run_structural_audit,
    )

    outputs = run_structural_audit(cfg)
    print(format_structural_summary(outputs))
    return EXIT_OK if outputs.valid else EXIT_AUDIT_FAILED


def _audit_media(cfg: DictConfig, args: argparse.Namespace) -> int:
    from conv_wm.data.audits.media_metadata import run_media_metadata_audit

    outputs = run_media_metadata_audit(cfg, max_workers=args.max_workers)
    summary = outputs.summary
    print(f"files: {summary['n_files']}  probe ok: {summary['n_probe_ok']}")
    print(f"summary: {outputs.summary_path}")
    return EXIT_OK


def _audit_video(cfg: DictConfig, args: argparse.Namespace) -> int:
    from conv_wm.data.audits.video_timeline import run_video_timeline_audit

    outputs = run_video_timeline_audit(cfg, max_workers=args.max_workers)
    summary = outputs.summary
    print(
        f"files: {summary['n_files']}  sampled consistent: "
        f"{summary['n_files_sampled_consistent']}  suspects: {summary['n_sampled_suspects']}"
    )
    print(f"summary: {outputs.summary_path}")
    return EXIT_OK


def _audit_audio(cfg: DictConfig, args: argparse.Namespace) -> int:
    from conv_wm.data.audits.audio_timeline import run_audio_timeline_audit

    outputs = run_audio_timeline_audit(cfg, max_workers=args.max_workers)
    summary = outputs.summary
    print(f"files: {summary['n_files']}  events: {summary['n_events']}")
    print(f"summary: {outputs.summary_path}")
    return EXIT_OK


def _audit_sync(cfg: DictConfig, _: argparse.Namespace) -> int:
    from conv_wm.data.audits.av_sync import run_av_sync_audit

    outputs = run_av_sync_audit(cfg)
    summary = outputs.summary
    print(f"zero start offsets: {summary['n_zero_start_offsets']}/{summary['n_files']}")
    print(f"summary: {outputs.summary_path}")
    return EXIT_OK


def _audit_annotations(cfg: DictConfig, _: argparse.Namespace) -> int:
    from conv_wm.data.audits.annotations import (
        format_annotation_summary,
        run_annotation_audit,
    )

    outputs = run_annotation_audit(cfg)
    print(format_annotation_summary(outputs))
    return EXIT_OK if outputs.valid else EXIT_AUDIT_FAILED


def _audit_vocal_annotation_coverage(cfg: DictConfig, args: argparse.Namespace) -> int:
    from conv_wm.data.audits.vocal_annotation_coverage import (
        run_vocal_annotation_coverage_audit,
    )

    command = ["conv-wm"]
    if args.config is not None:
        command += ["--config", str(args.config)]
    command += [
        "audit",
        "vocal-annotation-coverage",
        "--dataset",
        args.dataset,
    ]
    if args.coverage_config is not None:
        command += ["--coverage-config", str(args.coverage_config)]
    outputs = run_vocal_annotation_coverage_audit(
        cfg,
        dataset=args.dataset,
        coverage_config_path=args.coverage_config,
        command=" ".join(command),
    )
    statistics = outputs.report["coverage_statistics"]
    assert isinstance(statistics, dict)
    print(
        f"recordings: {statistics['successful_recording_count']}/"
        f"{statistics['recording_count']}  unresolved ratio: "
        f"{statistics['unresolved_identity_ratio']}"
    )
    print(f"report: {outputs.report_path}")
    return EXIT_OK


def _build_native_focal_voice_state(cfg: DictConfig, args: argparse.Namespace) -> int:
    from conv_wm.data.native_focal_voice_state import (
        format_outputs,
        run_native_focal_voice_state_build,
    )

    command = ["conv-wm"]
    if args.config is not None:
        command += ["--config", str(args.config)]
    command += ["build", "native-focal-voice-state", "--dataset", args.dataset]
    outputs = run_native_focal_voice_state_build(
        cfg, dataset=args.dataset, command=" ".join(command)
    )
    print(format_outputs(outputs))
    return EXIT_OK


def _build_control_focal_voice_state(cfg: DictConfig, args: argparse.Namespace) -> int:
    from conv_wm.data.control_focal_voice_state import (
        format_outputs,
        run_control_focal_voice_state_build,
    )

    command = ["conv-wm"]
    if args.config is not None:
        command += ["--config", str(args.config)]
    command += ["build", "control-focal-voice-state", "--dataset", args.dataset]
    outputs = run_control_focal_voice_state_build(
        cfg, dataset=args.dataset, command=" ".join(command)
    )
    print(format_outputs(outputs))
    return EXIT_OK


def _build_vocal_action_grid(cfg: DictConfig, args: argparse.Namespace) -> int:
    from conv_wm.data.vocal_action_grid import (
        format_outputs,
        run_vocal_action_grid_build,
    )

    command = ["conv-wm"]
    if args.config is not None:
        command += ["--config", str(args.config)]
    command += ["build", "vocal-action-grid", "--dataset", args.dataset]
    outputs = run_vocal_action_grid_build(
        cfg, dataset=args.dataset, command=" ".join(command)
    )
    print(format_outputs(outputs))
    return EXIT_OK


def _clean_annotations(cfg: DictConfig, _: argparse.Namespace) -> int:
    from conv_wm.data.cleaning import (
        format_annotation_cleaning_summary,
        run_annotation_cleaning,
    )

    outputs = run_annotation_cleaning(cfg)
    print(format_annotation_cleaning_summary(outputs))
    return EXIT_OK


AUDIT_COMMANDS: tuple[AuditCommand, ...] = (
    AuditCommand(
        "manifest", "inventory every file below the raw root", _audit_manifest
    ),
    AuditCommand(
        "structure", "validate dataset tables and relations", _audit_structure
    ),
    AuditCommand(
        "media",
        "probe container and stream metadata (needs manifest)",
        _audit_media,
        True,
    ),
    AuditCommand("video", "video timeline cadence (needs media)", _audit_video, True),
    AuditCommand(
        "audio",
        "audio packet and PCM-vs-PTS timelines (needs media)",
        _audit_audio,
        True,
    ),
    AuditCommand(
        "sync",
        "technical A/V alignment from stream metadata (needs media)",
        _audit_sync,
    ),
    AuditCommand(
        "annotations",
        "annotation integrity of registered datasets (needs media)",
        _audit_annotations,
    ),
    AuditCommand(
        "vocal-annotation-coverage",
        "measure acoustic and defensible focal voice annotation coverage",
        _audit_vocal_annotation_coverage,
        supports_dataset=True,
        supports_coverage_config=True,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for ``conv-wm``."""
    parser = argparse.ArgumentParser(
        prog="conv-wm",
        description="Reproducible data audits for the conversational world-model pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser(
        "audit", help="run a population audit and write its report"
    )
    audit_subparsers = audit.add_subparsers(dest="audit", required=True)
    for command in AUDIT_COMMANDS:
        sub = audit_subparsers.add_parser(command.name, help=command.help)
        if command.supports_workers:
            sub.add_argument(
                "--max-workers",
                type=int,
                default=4,
                help="parallel ffprobe workers (default 4)",
            )
        if command.supports_dataset:
            sub.add_argument(
                "--dataset",
                choices=("egocom", "ego4d", "all"),
                default="all",
                help="dataset population to audit (default all)",
            )
        if command.supports_coverage_config:
            sub.add_argument(
                "--coverage-config",
                type=Path,
                default=None,
                help="versioned VAD, identity and overlap configuration",
            )
        sub.set_defaults(handler=command.run)
    clean = subparsers.add_parser(
        "clean", help="derive source-faithful tables with approved transformations"
    )
    clean_subparsers = clean.add_subparsers(dest="clean", required=True)
    clean_annotations = clean_subparsers.add_parser(
        "annotations", help="clean registered annotation sources into interim tables"
    )
    clean_annotations.set_defaults(handler=_clean_annotations)
    build = subparsers.add_parser("build", help="build derived data products")
    build_subparsers = build.add_subparsers(dest="build", required=True)
    native_state = build_subparsers.add_parser(
        "native-focal-voice-state",
        help="derive the wearer's SPEAKING/SILENT/UNKNOWN timeline from native annotations",
    )
    native_state.add_argument(
        "--dataset",
        choices=("egocom", "ego4d", "all"),
        default="all",
        help="dataset to build (default all)",
    )
    native_state.set_defaults(handler=_build_native_focal_voice_state)
    control_state = build_subparsers.add_parser(
        "control-focal-voice-state",
        help="requantize the native vocal state at Δ by sub-step silence bridging",
    )
    control_state.add_argument(
        "--dataset",
        choices=("egocom", "ego4d", "all"),
        default="all",
        help="dataset to build (default all)",
    )
    control_state.set_defaults(handler=_build_control_focal_voice_state)
    action_grid = build_subparsers.add_parser(
        "vocal-action-grid",
        help="sample the control vocal state on the 100 ms grid as NO_EVENT/ONSET/OFFSET",
    )
    action_grid.add_argument(
        "--dataset",
        choices=("egocom", "ego4d", "all"),
        default="all",
        help="dataset to build (default all)",
    )
    action_grid.set_defaults(handler=_build_vocal_action_grid)
    datasets_parser = subparsers.add_parser("datasets", help="list registered datasets")
    datasets_parser.set_defaults(handler=_list_datasets)
    return parser


def _list_datasets(_: DictConfig, __: argparse.Namespace) -> int:
    from conv_wm.data import datasets

    for spec in datasets.iter_specs():
        tables = len(spec.structure.tables) if spec.structure else 0
        sources = len(spec.annotations.sources)
        print(
            f"{spec.name}: {spec.description} [{tables} tables, {sources} annotation sources]"
        )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        return int(args.handler(cfg, args))
    except (MissingPrerequisiteError, FFprobeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except (FileNotFoundError, ImportError, TypeError, ValueError, KeyError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
