# NeuroFisio Project Report

## What We Built

A complete clinical EEG research platform on the Temple University Hospital (TUH) EEG Corpus — the largest publicly available clinical EEG dataset (27,000+ hours, 15,000 patients, ~1.8 TB). In one week, we went from zero to a production-ready data pipeline, ML-ready datasets on HuggingFace, and a web-based EEG viewer.

---

## Infrastructure

| Component | Details |
|-----------|---------|
| **Compute** | spark-caeb.local — NVIDIA GB10, 128 GB unified RAM, 3.7 TB NVMe, 20 ARM cores |
| **Data** | 350 GB across 7 TUH EEG corpora (23,302 EDF files) |
| **Code** | [github.com/macayaven/neurofisio](https://github.com/macayaven/neurofisio) |
| **Datasets** | 12 HuggingFace datasets under [macayaven](https://huggingface.co/macayaven) org |
| **CI/CD** | GitHub Actions (ruff lint + format + pytest), runs in 15s |

---

## Data Assets

### Raw EEG Corpora (Downloaded)

| Corpus | Files | Size | Clinical Task | Annotations |
|--------|------:|-----:|---------------|-------------|
| **TUSL** | 112 | 1.6 GB | Seizure vs slowing | `.tse`, `.lbl` |
| **TUAR** | 310 | 5.4 GB | Artifact detection (5 types) | `.csv` per-channel |
| **TUEV** | 518 | 19 GB | 6-class event classification | `.rec`, `.lab` |
| **TUEP** | 2,821 | 36 GB | Epilepsy diagnosis | `.csv` clinical |
| **TUAB** | 2,993 | 59 GB | Normal vs abnormal EEG | Folder-based labels |
| **TUSZ** | 7,364 | 79 GB | Seizure detection (8+ types) | `.csv`, `.csv_bi` |
| **TUEG** | 9,184 | 150 GB | Unlabeled (pre-training) | None |
| **Total** | **23,302** | **~350 GB** | | |

### HuggingFace Datasets (Uploaded)

**Metadata datasets** (6 corpora) — file paths, annotations, subject demographics, recording metadata:

| Dataset | Status |
|---------|--------|
| `macayaven/tuh-eeg-tusl` | Uploaded |
| `macayaven/tuh-eeg-tuar` | Uploaded |
| `macayaven/tuh-eeg-tuev` | Uploaded |
| `macayaven/tuh-eeg-tuep` | Uploaded |
| `macayaven/tuh-eeg-tuab` | Uploaded |
| `macayaven/tuh-eeg-tusz` | Uploaded |

**Signal datasets** (windowed EEG waveforms — 22 channels x 2,500 samples @ 250 Hz, 10-second windows):

| Dataset | Windows | Shards | Status |
|---------|--------:|-------:|--------|
| `macayaven/tuh-eeg-tusl-signals` | 9,897 | 9 | Complete |
| `macayaven/tuh-eeg-tuar-signals` | 35,880 | 8 | Complete |
| `macayaven/tuh-eeg-tuev-signals` | 53,363 | 11 | Complete |
| `macayaven/tuh-eeg-tuep-signals` | 226,837 | 46 | Complete |
| `macayaven/tuh-eeg-tuab-signals` | 409,455 | 82 | Complete |
| `macayaven/tuh-eeg-tusz-signals` | 528,700 | 106 | Partial (~78 shards uploaded, failed during network outage) |
| **Total** | **~1.26M** | **262** | |

Each window is a `[22, 2500]` float32 tensor — the standard TCP bipolar montage used in clinical EEG worldwide.

---

## Software

### `neurofisio` Python Package

Reusable EEG processing library:

- **`parsers.py`** — TUH path parser, annotation loaders (CSV, TSE, REC, LAB formats)
- **`edf.py`** — EDF+ file reader, header inspector, corpus inventory builder
- **`montage.py`** — 22-channel TCP bipolar montage extraction (auto-detects reference naming)
- **`viz.py`** — Clinical-grade EEG segment renderer (dark theme, region coloring, annotation overlays)

### Streamlit EEG Explorer

Interactive web app for browsing clinical EEG recordings:
- Corpus/subject/session/file navigation
- Transport-bar playback with adjustable speed
- Per-channel TCP bipolar display with region coloring
- Annotation overlay visualization
- Dark clinical theme (JetBrains Mono + DM Sans)

### Dataset Builders

- **`create_hf_datasets.py`** — Metadata-level HuggingFace dataset builder
- **`create_hf_signal_datasets.py`** — Memory-safe signal extraction pipeline using generator-based processing and chunked parquet shard uploads (never exceeds ~5 GB RSS regardless of corpus size)

### CI/CD

- **Linting**: ruff (E/W/F/I/UP/B/SIM/RUF rules)
- **Formatting**: ruff format (line-length 99)
- **Testing**: pytest with smoke tests (package imports, constants, parser logic)
- **Local**: `uv run poe check` runs all 3 in sequence
- **Remote**: GitHub Actions on push/PR, completes in ~15s

---

## What Was Hard

### Memory-Safe Processing at Scale

The original signal extraction script accumulated all windowed EEG data in a Python list before creating the HuggingFace Dataset. Each 10-second window = 22 x 2,500 = 55,000 floats. For TUAB alone (~410K windows), this would need 750+ GB of RAM — far beyond the 128 GB available.

**Solution**: Rewrote the pipeline as Python generators that yield one row at a time, writing to parquet shard files every 5,000 rows with explicit `gc.collect()` and buffer clearing. Memory stays bounded at ~5 GB regardless of corpus size.

### Clinical EEG Standardization

TUH EEG files use varying channel naming conventions (`-REF`, `-LE`, mixed case, non-standard labels). The montage module auto-detects naming schemes and normalizes to the 22-channel TCP bipolar standard, handling 40+ electrode aliases.

---

## What AI Products Could Be Built

### Tier 1: Immediate Clinical Value (buildable now with existing data)

#### 1. NeuroTriage — AI EEG Triage System
**The flagship product.** A multi-stage pipeline that replicates the clinical neurologist's EEG reading workflow:

```
Raw EEG → Artifact Detection → Normal/Abnormal Triage → Event Classification → Report
             (TUAR)               (TUAB)                 (TUSZ + TUEV)
```

**Why it matters**: 30M EEGs/year in the US, only 18K neurologists. Interpretation backlogs of hours to days. Non-convulsive seizures (8-48% of ICU patients) go undetected without continuous monitoring. NCSE carries >50% mortality when treatment is delayed >1 hour.

**What exists**: Ceribell Clarity and Epitel REMI do seizure detection only. No system does the full workflow: quality gate, triage, event classification, and reporting.

**Moat**: We have all 6 annotated sub-corpora in a unified pipeline — every other team trains on a single task.

**Revenue model**: SaaS per-read ($15-30/EEG), or per-bed ICU monitoring subscription ($500-2K/month).

---

#### 2. EEG Foundation Model API
**A general-purpose EEG encoder** pre-trained on the 27K-hour TUEG corpus, offered as an API or fine-tunable model.

**Why**: Every EEG AI startup re-trains from scratch. A strong foundation model (like BERT for NLP) would let downstream developers fine-tune with 100x less data.

**Differentiation from Neuro-GPT/LaBraM**: Those are research artifacts. We'd offer a production API with guaranteed SLAs, fine-tuning endpoints, and clinical-grade preprocessing (the neurofisio package handles the gnarly EDF parsing and montage normalization).

**Revenue model**: API calls ($0.01/inference), fine-tuning jobs ($50-500), enterprise licenses.

---

#### 3. Automated EEG Report Generator
**Given a raw EEG, produce a structured clinical report** following standard neurology reporting format.

This is the most time-consuming part of a neurologist's workflow. Even if the AI just drafts the report for human review, it saves 15-20 minutes per read.

**Approach**: Fine-tune a language model on the structured outputs from the multi-task pipeline (artifact flags, normal/abnormal, events detected, temporal patterns). The TUEP metadata provides epilepsy risk context.

**Revenue model**: Per-report ($5-15), integrated into EHR systems.

---

### Tier 2: Adjacent Markets (require additional data or partnerships)

#### 4. Neonatal EEG Monitoring
Neonatal seizures are subtle and missed >75% of the time by bedside nurses. A specialized model trained on pediatric EEG could be transformative. Would require neonatal EEG training data (not in TUH corpus).

#### 5. Sleep Stage Classification
The same 22-channel EEG setup is used in polysomnography. A sleep staging model could power at-home sleep study analysis. Would complement consumer devices (Oura, WHOOP) that lack clinical-grade EEG.

#### 6. Brain-Computer Interface (BCI) Signal Processing
The EEG preprocessing pipeline (artifact rejection, montage normalization) is directly applicable to BCI applications. An SDK for BCI developers could leverage the neurofisio package.

#### 7. Drug Trial EEG Biomarker Platform
Pharmaceutical companies use EEG as a biomarker in neurological drug trials (epilepsy, Alzheimer's, Parkinson's). An automated analysis platform could standardize EEG endpoints across multi-site trials, reducing variability and cost.

---

### Tier 3: Platform Plays

#### 8. Clinical EEG Data Marketplace
Curated, preprocessed, ML-ready EEG datasets for researchers and startups. The signal datasets on HuggingFace are a proof of concept. A premium tier could offer larger datasets, custom preprocessing, and compliance documentation.

#### 9. EEG Education Platform
Interactive EEG learning tool for neurology residents and EEG technicians. The Streamlit explorer already demonstrates core functionality. Add quiz modes, pattern recognition training, and case-based learning.

---

## Recommendation: Where to Start

| Priority | Product | Time to MVP | Data Ready? | Revenue Potential |
|----------|---------|-------------|-------------|-------------------|
| **P0** | TUAB binary classifier (normal/abnormal) | 2 weeks | Yes | Validates pipeline |
| **P1** | Foundation encoder (self-supervised on TUEG) | 4 weeks | Yes | Enables all downstream |
| **P2** | Seizure detection (TUSZ + TUAR) | 3 weeks | Yes (TUSZ upload needs retry) | High clinical demand |
| **P3** | Multi-task pipeline (NeuroTriage v1) | 3 weeks | Yes | Differentiated product |
| **P4** | Report generation | 4 weeks | Partial | Highest per-unit value |

**Start with P0**: A binary normal/abnormal EEG classifier trained on TUAB (409K labeled windows, train/eval splits ready) is the fastest path to a working model. It validates the entire pipeline — data loading, training, inference — and has standalone clinical value as a screening tool.

**The strategic bet is P1**: A foundation model pre-trained on 27K hours of unlabeled EEG would be the first production-grade EEG encoder. It makes P2-P4 dramatically easier and creates a durable competitive advantage.

---

## Current State Summary

```
Data Pipeline:    ████████████████████████████████████████ 100% (350 GB downloaded)
HF Metadata:      ████████████████████████████████████████ 100% (6/6 corpora)
HF Signals:       █████████████████████████████████████░░░  92% (5/6 complete, TUSZ partial)
Code + CI/CD:     ████████████████████████████████████████ 100% (GitHub + Actions)
EEG Viewer:       ████████████████████████████████████████ 100% (Streamlit app)
Model Training:   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0% (next phase)
```

The foundation is complete. The data is downloaded, preprocessed, and on HuggingFace. The code is versioned and CI'd. The next step is training.
