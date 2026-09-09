# Raw Dataset Manifest

## Objective

Build a deterministic inventory of all files available under the raw data root.

The manifest does not modify or validate the scientific content of the data.
Its role is to describe what raw files are available before any processing step.

## Input

Global raw data root:

```text
${paths.raw}
```

Expected structure:
raw/
├── EgoCom/
├── Ego4D/
└── <future datasets>/

The first directory below raw/ is interpreted as the dataset name.

## Output

${paths.reports}/manifest/raw_manifest.parquet

## Schema

| Column               | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `dataset`            | Dataset inferred from the first directory below the raw root |
| `relative_path`      | Path relative to the global raw root                         |
| `file_name`          | File basename                                                |
| `extension`          | Normalized lowercase file extension                          |
| `file_type`          | Broad physical format family                                 |
| `size_bytes`         | File size in bytes                                           |
| `checksum`           | Optional SHA-256 checksum                                    |
| `checksum_algorithm` | Checksum algorithm, when enabled                             |

## File type classification

The manifest performs only broad physical classification:

- video
- audio
- tabular
- structured_data
- other

Semantic roles such as transcript, gaze_annotation, or
metadata are intentionally not inferred at this stage.

## Exclusions

Known operating-system artifacts are ignored, including:

- .DS_Store
- AppleDouble files (.\_\*)
- Thumbs.db
- \_\_MACOSX

Legitimate but unknown dataset files are retained as other.

## Determinism

Files are sorted by:

1. dataset
2. relative path

The same raw filesystem state should therefore produce the same logical
manifest.

## Validation

The following invariants are checked:

- all required columns exist;
- (dataset, relative_path) is unique;
- file sizes are non-negative;
- relative paths are non-empty.

## Checksums

SHA-256 checksums are optional because hashing large video collections is
I/O intensive.
Checksums are disabled by default and can be enabled through configuration.

## Implementation

Core implementation:

```code
src/conv_wm/data/manifest.py
```

Hydra entry point:

```code
src/conv_wm/data/build_manifest.py
```

Tests:

```code
tests/data/test_manifest.py
tests/data/test_build_manifest.py
```

## Design decisions

The manifest is intentionally limited to filesystem-level information.
Media properties such as FPS, duration, codec, sample rate, and timestamps
belong to the media audit stage rather than the manifest stage.
Dataset-specific semantics are handled later by dataset-specific processing
or adapters.
