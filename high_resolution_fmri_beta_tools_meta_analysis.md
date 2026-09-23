# Beta-estimation software in high-resolution task fMRI: rapid meta-analysis

Search completed: 3 August 2026.

## Bottom line

Across **25 independent high-resolution task-fMRI acquisitions**, the most common primary beta-estimation family was **SPM (8/25; 32.0%)**. Custom, in-house, or insufficiently named GLMs accounted for **4/25 (16.0%)**. FSL and AFNI each accounted for **3/25 (12.0%)**, BrainVoyager for **2/25 (8.0%)**, and FS-FAST, Nistats/Nilearn, mrTools, GLMdenoise, and nideconv each accounted for **1/25 (4.0%)**.

These are descriptive study proportions, not participant-weighted estimates. Wilson 95% confidence intervals are shown because the corpus is small.

| Primary beta-estimation family | Independent acquisitions | Percentage | Wilson 95% CI |
|---|---:|---:|---:|
| SPM (SPM8/SPM12; including MarsBaR or Nipype wrappers) | 8 | 32.0% | 17.2–51.6% |
| Custom / in-house / implementation not adequately named | 4 | 16.0% | 6.4–34.7% |
| FSL (FILM/FEAT) | 3 | 12.0% | 4.2–30.0% |
| AFNI | 3 | 12.0% | 4.2–30.0% |
| BrainVoyager | 2 | 8.0% | 2.2–25.0% |
| FS-FAST | 1 | 4.0% | 0.7–19.5% |
| Nistats/Nilearn | 1 | 4.0% | 0.7–19.5% |
| mrTools | 1 | 4.0% | 0.7–19.5% |
| GLMdenoise + fractional ridge regression | 1 | 4.0% | 0.7–19.5% |
| nideconv | 1 | 4.0% | 0.7–19.5% |
| **Total** | **25** | **100.0%** | — |

### Paper-level sensitivity analysis

One included paper reused the Natural Scenes Dataset (NSD). Counting that secondary paper separately gives 26 papers: SPM 26.9%, custom/unclear 15.4%, FSL 11.5%, AFNI 11.5%, BrainVoyager 7.7%, GLMdenoise 7.7%, and FS-FAST, Nistats/Nilearn, mrTools, and nideconv 3.8% each. The qualitative ranking is unchanged. The independent-acquisition result above is preferable because it prevents a heavily reused dataset from receiving extra weight.

## Eligibility and coding rules

This was a rapid, reproducible evidence synthesis of open, full-text studies located through Europe PMC full-text searches and backward/forward reference checking.

Included studies had to:

1. contain a primary human task-fMRI experiment;
2. acquire native functional data with every reported voxel dimension at or below 1.8 mm, or contain an explicitly high-resolution/submillimeter task-fMRI arm;
3. estimate task-related GLM beta coefficients or parameter estimates; and
4. provide enough methods information to classify the beta-fitting implementation, with an explicit “custom/unclear” category retained when reporting was insufficient.

Excluded were resting-state-only studies, papers in which only the anatomical image was high resolution, ordinary-resolution fMRI merely resampled to ≤1.8 mm, native functional acquisitions above the cutoff, nonhuman-only studies, reviews/protocols, and studies whose primary response estimate was Fourier amplitude or block averaging rather than a beta coefficient.

The coded variable is the package that **fit the task GLM**, not every package in the pipeline. Thus, fMRIPrep, FreeSurfer, ANTs, FSL TOPUP, AFNI motion correction, or SPM preprocessing were not counted unless the methods tied that package to beta estimation. Wrappers were assigned to the underlying estimator where clear (for example, SPM through Nipype). Each acquisition was assigned one primary family so percentages sum to 100%; secondary beta-fitting tools are preserved in the CSV notes.

## Included independent acquisitions

| Year | Study | Native functional resolution | Primary beta tool/family |
|---:|---|---|---|
| 2011 | [Hutton et al.](https://doi.org/10.1016/j.neuroimage.2011.04.018) | 1.1 × 1.1 × 1.8 mm | SPM8 |
| 2013 | [De Martino et al.](https://doi.org/10.1371/journal.pone.0060514) | 1.0 and 0.8 mm isotropic | BrainVoyager |
| 2013 | [Lutti et al.](https://doi.org/10.1002/mrm.24398) | 1.5 mm isotropic | SPM8 |
| 2016 | [Schallmo et al.](https://doi.org/10.1167/16.10.19) | 1.2 mm isotropic | AFNI 3dDeconvolve |
| 2017 | [Tootell & Nasr](https://doi.org/10.1523/jneurosci.0690-17.2017) | 1.0 mm isotropic | FS-FAST |
| 2018 | [Pinho et al.](https://doi.org/10.1038/sdata.2018.105) | 1.5 mm isotropic | Nistats/Nilearn |
| 2019 | [Hindy et al.](https://doi.org/10.1038/s41467-019-12016-9) | 1.5 mm isotropic | FSL FILM |
| 2020 | [Vizioli et al.](https://doi.org/10.1038/s41598-020-64044-x) | 0.8 mm isotropic | Custom least-squares GLM |
| 2020 | [Tabas et al.](https://doi.org/10.7554/eLife.64501) | 1.5 mm isotropic | SPM12 via Nipype |
| 2021 | [Allen et al.](https://doi.org/10.1038/s41593-021-00962-x) | 1.8 mm isotropic | GLMdenoise + fractional ridge regression |
| 2022 | [Saadon-Grosman et al.](https://doi.org/10.1152/jn.00165.2022) | 1.8-mm discovery arm | FSL FEAT |
| 2022 | [Liu et al.](https://doi.org/10.1038/s41467-022-33580-7) | 1.2-mm BOLD; 0.82-mm VASO | mrTools |
| 2022 | [Stein et al.](https://doi.org/10.1093/texcom/tgac047) | 1.5 and 1.75 mm isotropic | SPM12 EstimateModel |
| 2023 | [Lankinen et al.](https://doi.org/10.1002/hbm.26046) | 1.0 mm isotropic | In-house MATLAB GLM |
| 2024 | [Guo et al.](https://doi.org/10.1371/journal.pbio.3002375) | 1.5 mm isotropic | AFNI/mripy |
| 2024 | [Czajko et al.](https://doi.org/10.1038/s41467-024-44810-5) | 1.5 mm isotropic | SPM12 |
| 2024 | [Ara et al.](https://doi.org/10.1093/cercor/bhae316) | 1.5 mm isotropic | SPM + MarsBaR |
| 2024 | [Lu et al.](https://doi.org/10.1038/s41467-024-53968-x) | 1.2-mm human 7T experiment | SPM12 |
| 2025 | [Müller-Axt et al.](https://doi.org/10.1093/brain/awae235) | 1.25 × 1.25 × 1.2 mm | SPM12 |
| 2025 | [Pizzuti et al.](https://doi.org/10.1007/s00429-025-02906-8) | 0.8-mm task; 1.8-mm localizer | BrainVoyager |
| 2024 | [Yang et al.](https://doi.org/10.1162/imag_a_00404) | 1.5 mm isotropic | Implementation not reported |
| 2025 | [Lloyd et al.](https://doi.org/10.1093/cercor/bhaf101) | 1.5 mm isotropic | nideconv (primary ROI analysis) |
| 2025 | [Trutti et al.](https://doi.org/10.7554/eLife.97874) | 1.5 mm isotropic | FSL FILM/FEAT |
| 2025 | [Lin et al.](https://doi.org/10.1038/s42003-025-08871-6) | 1.5 mm isotropic | AFNI GLM |
| 2026 | [Priestley et al.](https://doi.org/10.1038/s41467-026-68349-9) | 1.5 mm isotropic | Custom OLS (primary ROI analysis) |

Secondary NSD paper used only in the paper-level sensitivity analysis: [Guest et al. (2025)](https://doi.org/10.1038/s41467-025-67472-3), which used the NSD 1.8-mm acquisition and its GLMdenoise/ridge trial estimates.

## Interpretation

The main result is that conventional neuroimaging suites still dominate beta estimation in high-resolution task fMRI: SPM, FSL, AFNI, BrainVoyager, and FS-FAST together account for **17/25 studies (68%)**. Specialized or code-centric estimators—Nistats/Nilearn, mrTools, GLMdenoise, nideconv, and custom/unclear implementations—account for **8/25 (32%)**.

This is best interpreted as a snapshot of reported practice, not proof that one estimator performs better. Tool choice is confounded with laboratory, scientific domain, acquisition type, design (block, event-related, single-trial), and whether inference is voxelwise or ROI-based. The broad confidence intervals also show that fine-grained rankings beyond SPM are unstable in a corpus of 25 acquisitions.

## Limitations

- This is a rapid open-full-text review, not a preregistered systematic review across every bibliographic database and subscription-only paper.
- “High resolution” has no universal cutoff. The operational cutoff here follows the requested 1.8-mm threshold and requires native functional—not anatomical or resampled—resolution.
- The sample contains heterogeneous 3 T/7 T, BOLD/VASO, whole-brain/partial-volume, block/event-related, and ROI/voxelwise analyses.
- Package reporting was sometimes incomplete. Four studies were therefore retained in a custom/unclear category instead of being guessed into a named package.
- The proportions are unweighted by participant count because the target is methods prevalence by study/acquisition.
