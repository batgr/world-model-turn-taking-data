# Data Pipeline

This directory documents the data pipeline used for the conversational world model project.

## Core principles

1. Raw data is immutable.
2. Every derived artifact must be reproducible from raw data.
3. Data quality checks are separated from data correction.
4. Temporal alignment relies on timestamps rather than frame indices whenever possible.
5. Dataset-specific processing is isolated from generic processing logic.
6. Notebooks are used for exploration and analysis; reusable logic lives in `src/`.
7. Scientific or methodological decisions must be documented and, when possible, supported by references.
8. Model-specific transformations such as resampling, windowing, and feature extraction are applied as late as possible.

## Pipeline stages

raw → interim → validated → processed → model_ready
