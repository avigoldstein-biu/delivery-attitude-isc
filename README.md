# Analysis code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22808428.svg)](https://doi.org/10.5281/zenodo.22808428)

Code to reproduce the analyses reported in *Delivery and Attitude Modulate Shared Neural Processing of Speech Through Distinct Frequency Channels* (Atia, Berson &
Goldstein).

## Requirements

MATLAB R2020b with FieldTrip (source reconstruction and surface rendering) and Python 3.13
with NumPy, SciPy, librosa, soundfile, pandas, matplotlib, Pillow, FOOOF/specparam, MNE and FFmpeg
(via `imageio-ffmpeg`). The word-onset control additionally uses `faster-whisper`,
`transformers` and `torchaudio`.

Paths are constants at the top of each script. `ROOT` is the data root; `DERIV` is the
working directory for everything the pipeline produces -- the source reconstructions, the
surface renders, the forced-alignment output and the intermediate result files. In this copy
`DERIV` is `<root>/reanalysis`.

The scripts expect the following inputs:

    <root>/Charisma/char_*/           per-participant Study 1 MEG, head model, trigger info
    <root>/iscex/ISCex_*/             per-participant Study 2 MEG
    <root>/*.MPG, *.wmv               the stimulus clips
    <root>/quest.xlsx                 Study 1 questionnaire scores
    <root>/iscex/study23_labeled.csv  Study 2 questionnaire scores

Two of these are produced upstream of this folder rather than by it.

**MEG cleaning.** The Study 1 MEG file each script reads is named `hb,xc,lf_c,rfhp0.1Hz`,
where `hb` and `xc` record that heartbeat and external-noise cancellation (Tal & Abeles,
2013) have already been applied. That step is upstream of this repository and no script here
performs or calls it. It was done with cleanMEG_BIU
(https://github.com/yuval-harpaz/cleanMEG_BIU, commit `1710f63`, 2016-05-20), which is
distributed under GPL v2 and includes the `@pdf4D` class for reading 4D/BTi recordings. It is
needed only to rebuild the cleaned files from the raw recordings, and those are in the
restricted tier of the data deposit -- everything the open tier supports begins downstream of
it. Ocular components are removed by `matlab/ft_ica_full3.m`.

**Head models.** `headmodelN.mat` and `sourcemodel.mat` are per-participant inputs, built
from the individual anatomy at an earlier stage. They were created individually with script in `matlab/coregistration.m`. The
reconstruction scripts load them and compute the leadfield from them. Reproducing the
analyses therefore requires these files.

The reconstruction scripts write to `source_myica3/` (Study 1, band-limited),
`source_bb_ica/` (Study 1, broadband) and the Study 2 equivalent, all under `DERIV`. Every
analysis script reads from those rather than from the raw recordings; the exception is
`study1_sensor_extract.py`, which goes back to the raw MEG for the sensor-level envelope
analysis.

## Order

Source reconstruction must run first; everything else reads its output.

### 1. Source reconstruction (MATLAB)

| Script | Produces |
|---|---|
| `matlab/ft_ica_full3.m` | Study 1, five bands, all three conditions, objective ocular ICA |
| `matlab/ft_source_broadband_ica.m` | Study 1, broadband, for the power analyses |
| `matlab/source_reconstruction_iscex.m` | Study 2, both tasks |

One ICA solution is computed per participant from all conditions concatenated and applied to
each, so the ocular correction is constant within a participant.

### 2. Stimulus description and feature extraction

| Script | Produces |
|---|---|
| `stimulus_acoustic_features.py` | Table S1 (computed at matched presentation level) |
| `stimulus_visual_features.py` | Table S2 |
| `align_speech.py` | word onsets by ASR + CTC forced alignment, for the word-onset control |
| `study2_build_features.py` | the ten-feature battery for the Study 2 stimuli |

The Study 1 feature battery (Table S3) is built by `build_features()` in `lib.py`, called by
the scripts that use it and cached to `_feat_cache.npz`.

### 3. Study 1

| Script | Produces |
|---|---|
| `study1_isc.py` | ISC at each control stage; Table S5, laterality, word-onset control |
| `study1_behaviour.py` | manipulation check and questionnaire comparisons (Section 3.3) |
| `isc_within_condition_null.py` | within-condition significance of the ISC itself, both studies (Section 2.4) |
| `study1_anova.py` | three-condition ANOVA and contrasts against silence; Table S9 |
| `study1_power_anova.py` | spectral power across conditions; Table S10 |
| `study1_power_specparam.py` | regional alpha power and the periodic/aperiodic split |
| `study1_feature_families.py` | which feature family carries the reduction; Table S6 |
| `study1_visual_fine_control.py` | extended optic-flow visual model; Supplement S2.6 |
| `study1_sensor_extract.py` → `study1_envelope_control.py` | envelope–MEG coupling; Table S4 |
| `study1_regression.py` | ISC predicted from the questionnaire battery |
| `study1_brain_behavior.py` | ISC–behaviour correlations by band and region set |
| `study1_is_rsa.py` | inter-subject representational similarity; Supplement S7 |


### 4. Study 2

| Script | Produces |
|---|---|
| `study2_isc_stats.py` | attitude and audience-cue contrasts by band and region; Table S8 |
| `study2_feature_regression.py` | low-level feature control, with the group tests at each stage |
| `study2_power_specparam.py` | spectral power across groups, and the periodic/aperiodic split |
| `study2_periodic_control.py` | ISC recomputed on the periodic component |
| `study2_trait_synchrony.py` | cross-task synchronisation trait; Figure 5 |
| `study2_behaviour.py` | manipulation check, ISC-behaviour regression, personality control |
| `study2_carryover.py` | were the two crossed manipulations inert on each other's task? |

`study2_isc_stats.py` runs first within this section: it writes `<DERIV>/study2_isc_results.npz`
(per-region effect, two-sided p and one-sided p for each band and each manipulation), which
`cross_study_convergence.py`, `roi_contiguity_null.py` and `fig_export_render_stats.py` all read.

### 5. Cross-study

`cross_study_convergence.py` — the region-level conjunction, laterality, and the raw-stage
enrichment within the anatomically pre-specified right auditory-perisylvian ROI. The
stage-resolved version of that enrichment test, which is what Table S7 reports, is in
`roi_stage_grid.py` below.

`roi_stage_grid.py` — the ROI enrichment test at every stage of control, for both studies.
Computes the Study 2 effect maps for both manipulations, raw, after feature regression, and after
both controls (cached to `<DERIV>/s23_stage_maps.npz`), then enumerates all 1,028,790 four-parcel
sets for an exact p. Produces Table S7. Its docstring records why Study 1 is reported after both
controls and Study 2 after feature regression.

`roi_against_zero.py` — the OTHER region-of-interest test of Section 2.6. Where `roi_stage_grid.py`
asks whether an effect is more concentrated in the four parcels than elsewhere in the atlas, this
asks whether the region-averaged group difference departs from zero: each participant contributes
their leave-one-out correlation averaged across the four parcels, and the group labels are permuted
as in Section 2.5. Both contrasts at all three control stages, so the grid matches Table S7. Checks
its own staging against `s23_stage_maps.npz`, so run `roi_stage_grid.py` first.

`roi_contiguity_null.py` — repeats the ROI enrichment test against a null restricted to connected
four-parcel sets, so that the comparison sets match the ROI in spatial contiguity. All such sets
are enumerated, so the p values are exact. Reads the reported-stage maps from
`roi_stage_grid.py`. Produces Table S7b and caches the 72-region adjacency matrix to
`<DERIV>/roi_adjacency.npz`.

### 6. Figures

| Script | Produces |
|---|---|
| `fig_power_spectrum.py` | Figure 2A |
| `fig_global_isc.py` | Figure 2B |
| `fig_export_render_stats.py` → `matlab/render_patch.m` → `fig_brain_montage.py` | Figures 3 and 4A |
| `fig_global_isc_s23.py` | Figure 4B |
| `fig_trait_synchrony.py` | Figure 5 |
| `fig_convergence_export.py` → `matlab/render_convergence.m` → `fig_convergence.py` | Figure 6 |
| `build_figures.py` | Figure 1, and the numbered set assembled from the panels above |

`build_figures.py` runs last: it draws the design schematic, composites the two-panel figures,
stamps everything at 300 dpi and reports each figure's printed width against the 80-180 mm limit.
`fig_trait_synchrony.py` reads the values `study2_trait_synchrony.py` writes to
`<DERIV>/trait_synchrony.npz` rather than recomputing or transcribing them.

`render_patch.m` samples the atlas at each surface vertex directly. Panels are masked by the
significance of the stage they depict, so only regions surviving FDR correction at that stage
are painted.

`render_convergence.m` renders the Figure 6 maps unmasked on one diverging colour scale, with the
four ROI parcels outlined by tracing every mesh edge that straddles the ROI boundary. The three
maps are unthresholded deliberately: the claim is that two manipulations land on the same patch
and a third does not, and masking each map by its own FDR result would instead give three
differently-thresholded blob patterns with the cue map blank. `fig_convergence.py` assembles that
with the enrichment null distributions, which are the reference sets the Table S7 p values come
from.

## Conventions

Condition contrasts are tested at the participant level, one leave-one-out ISC value per
participant per region and band, and FDR-corrected across the 72 regions within each band.
Study 1 uses a paired test, Study 2 a group-label permutation null of 10,000 permutations
(`lib2.NPERM`) in which the leave-one-out ISC is recomputed within the permuted groups. Permuting
frozen subject-level values instead treats them as exchangeable when they are not, and returns p
values several times too small. Study 2 results are reported both two-sided and one-sided, since
the direction of the attitude effect was predicted and that of the audience cue was not.

Phase-scrambled feature nulls are averaged over 20 draws: a single draw varies by roughly
twenty percentage points of retained effect, enough to reverse the apparent sign of a small
reduction.

## Licence and citation

MIT (see `LICENSE`). The licence requires only that the copyright notice be retained; it does not
require citation.

If you use this code, please cite the paper. `CITATION.cff` carries the details, and GitHub and
Zenodo both read it, so the "Cite this repository" button and the archived release produce a
correct reference without anyone transcribing it.

This code is archived at https://doi.org/10.5281/zenodo.22808428, which always resolves to the
most recent version; each release also has its own DOI if you need to pin one. The data it
reads are deposited separately at https://doi.org/10.5281/zenodo.22808367.

MEG cleaning is not covered by this licence: it is done by cleanMEG_BIU, which is separately
distributed under GPL v2 (see **MEG cleaning** above). No code from it is included here.

