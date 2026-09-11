# v1.0.0 — JRSI resubmission analysis

Release date: 2026-09-10

This release freezes the analysis supporting the resubmission of manuscript rsif-2026-0121, "Testing cortical-state persistence as a dynamical mechanism of freezing of gait".

## Scope

The release reproduces the final analyses from the public Mendeley Data version-3 filtered dataset (DOI: 10.17632/r8gmbtv7w2.3):

- full-grid within-recording circular-shift tests for concurrent EEG/FoG spectral contrasts;
- movement trimming and movement-residualization sensitivities;
- common-criterion comparison of continuous, burst, and latent-state representations;
- EEG-only latent-state persistence analysis with recording-bounded runs and prospective FoG-onset risk;
- small-cluster participant-level inference and robustness across K=2--5, alternative risk intervals, link functions, first-onset restriction, and leave-one-participant-out refits;
- strictly antecedent non-overlapping theta/low-beta analyses over 3-, 5-, and 10-s intervals;
- leave-one-participant-out predictive validation with preprocessing estimated from training participants only.

## Frozen interpretation

The final results support heterogeneous concurrent and temporally aligned pre-onset cortical spectral structure, but do not support the hypothesized increase in FoG-onset risk with longer occupancy of EEG-defined states. Strictly antecedent spectral structure does not yield robust participant-held-out predictive improvement.

## Reproducibility

Use `run_all.sh` with the extracted public `Filtered Data` directory. The workflow and `requirements.txt` define the reproducible environment. Compact manuscript-supporting tables are stored under `frozen_results/`.

## Archival release

This repository is prepared for archival as GitHub release `v1.0.0` and Zenodo ingestion. After Zenodo mints the DOI, update the README and manuscript data-accessibility statement with the version DOI.
