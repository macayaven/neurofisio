# NeuroTriage: AI-Powered Clinical EEG Interpretation Pipeline

## The Problem

A neurologist takes 20-30 minutes to read a single EEG. In the US alone, there are **~30 million EEGs recorded annually** but only ~18,000 practicing neurologists. The result:

- **EEG interpretation backlogs** of hours to days in most hospitals
- **8-48% of ICU patients** have non-convulsive seizures that go undetected without continuous EEG monitoring
- **Non-convulsive status epilepticus (NCSE)** carries >50% mortality when treatment is delayed >1 hour
- **Rural and community hospitals** often lack 24/7 neurologist coverage for urgent EEG reads

Existing FDA-cleared AI tools (Ceribell Clarity, Epitel REMI Vigilenz) address only seizure detection. No commercially available system performs the **full clinical EEG interpretation workflow**: signal quality assessment, normal/abnormal triage, event identification, and structured reporting.

## The Solution

**NeuroTriage** is a multi-stage AI pipeline that replicates the clinical neurologist's EEG reading workflow:

```
Raw EEG → [Artifact Detection] → [Normal/Abnormal Triage] → [Event Classification] → [Structured Report]
              TUAR data              TUAB data               TUSZ + TUEV data         All corpora
```

**Stage 1 - Signal Quality Gate:** Identifies and flags artifact-contaminated segments (eye movement, muscle, electrode) so downstream models operate on clean signal. Trained on TUAR's 160,073 per-channel artifact annotations.

**Stage 2 - Triage:** Binary classification of the overall recording as clinically normal or abnormal. Trained on TUAB's 2,993 expert-labeled recordings with 99% inter-rater agreement. This is the **screening gate** - normal EEGs are deprioritized, abnormals routed to detailed analysis.

**Stage 3 - Clinical Event Detection:** Multi-label identification of seizures (8+ types), epileptiform discharges (spike-wave, GPED, PLED), and pathological slowing. Trained on TUSZ (7,364 files, 3,964 seizure events) and TUEV (518 files, 6 event classes). TUSL data teaches the model to distinguish seizures from slowing - a common source of false positives.

**Stage 4 - Structured Report Generation:** Synthesizes per-channel findings into a clinical summary following standard EEG reporting format: technical quality, background activity, abnormal findings, and clinical correlation. Epilepsy risk contextualized using TUEP patient metadata.

## Technical Approach

### Foundation Model Pre-Training

Self-supervised pre-training on the **full 27,074-hour TUEG corpus** (69,670 unlabeled EDF files) using masked channel-time reconstruction. This follows the approach validated by Neuro-GPT and EEGFormer, but with a key advantage: we fine-tune sequentially on **all 6 annotated sub-corpora** rather than a single downstream task.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│              EEG Foundation Encoder                  │
│  (Transformer, pre-trained on 27K hours of TUEG)    │
│  Input: 22-channel TCP montage, 250 Hz, 10s windows │
└──────────────┬──────────────────────────────────────┘
               │ shared representations
    ┌──────────┼──────────┬──────────┬──────────┐
    ▼          ▼          ▼          ▼          ▼
┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
│Artifact││ Triage ││Seizure ││ Event  ││Slowing │
│  Head  ││  Head  ││  Head  ││  Head  ││  Head  │
│ (TUAR) ││ (TUAB) ││ (TUSZ) ││ (TUEV) ││ (TUSL) │
└────────┘└────────┘└────────┘└────────┘└────────┘
```

**Multi-task learning** with shared encoder and task-specific classification heads. Each head is fine-tuned on its respective corpus.

### Hardware

- NVIDIA GB10 (128 GB unified memory) - sufficient for training transformer encoders on EEG
- 3.7 TB NVMe storage, 20 ARM cores
- Full TUH EEG corpus locally available (~1.8 TB)

## Development Phases

| Phase | Duration | Data | Deliverable | Clinical Gate |
|-------|----------|------|-------------|---------------|
| **P0: Baseline** | 2 weeks | TUAB (58 GB) | CNN binary classifier, normal vs abnormal | AUC > 0.85 |
| **P1: Foundation** | 4 weeks | TUEG subset (200 GB) | Self-supervised encoder, transfer to TUAB | AUC > 0.90 on TUAB |
| **P2: Seizure** | 3 weeks | TUSZ (81 GB) + TUAR (5.4 GB) | Seizure detection + artifact rejection | Sensitivity > 0.80, FA < 1/24h |
| **P3: Multi-task** | 3 weeks | TUEV + TUSL | Add event classification + slowing heads | 6-class acc > 0.75 |
| **P4: Report Gen** | 4 weeks | TUEP metadata + all | Structured report from pipeline output | Neurologist validation |

**P0 is the MVP** - a binary normal/abnormal screener already has clinical value and validates the pipeline.

## Competitive Positioning

| Feature | Ceribell Clarity | Epitel REMI | **NeuroTriage** |
|---------|-----------------|-------------|-----------------|
| Seizure detection | Yes | Yes | Yes |
| Normal/abnormal triage | No | No | **Yes** |
| Artifact handling | Basic | Basic | **Per-channel, 5 types** |
| Event classification | No | No | **6 classes** |
| Slowing differentiation | No | No | **Yes** |
| Structured reporting | No | No | **Yes** |
| Epilepsy risk context | No | No | **Yes** |
| Hardware requirement | Proprietary headband | Proprietary patches | **Standard clinical EEG** |
| Foundation model | No | No | **Yes (27K-hour pre-train)** |

## Clinical Value Proposition

**For Emergency Departments:**
Instant EEG triage → abnormal results flagged within minutes instead of hours. Reduces missed NCSE.

**For ICU Continuous Monitoring:**
Automated real-time analysis of continuous EEG. Alerts for seizures, status changes, and evolving abnormalities.

**For Community/Rural Hospitals:**
AI-first read with structured report, enabling remote neurologist confirmation instead of full interpretation.

**For Neurologists:**
Pre-filtered, pre-analyzed queue. Normal EEGs auto-cleared. Attention focused on abnormal findings with AI-highlighted regions of interest.

## Metrics for Success

| Metric | Target | Benchmark |
|--------|--------|-----------|
| Normal/Abnormal AUC | > 0.92 | Lopez 2017 reported 0.88 |
| Seizure sensitivity | > 0.85 | Ceribell Clarity: ~0.90 |
| Seizure false alarm rate | < 1 per 24h | Clinical acceptability threshold |
| Event classification accuracy | > 0.78 | Harati 2015 baseline |
| Time-to-result | < 60 seconds | Current: 20-30 min (human) |
| Pipeline end-to-end latency | < 5 seconds per 10s window | Enables real-time monitoring |

## Regulatory Pathway

NeuroTriage is a **Clinical Decision Support (CDS) tool** - it assists but does not replace the neurologist. Initial deployment as a triage/prioritization system, which falls under FDA's more permissive CDS guidance (not requiring 510(k) for initial clinical evaluation).

For eventual FDA clearance: the 510(k) pathway with predicate devices (Ceribell Clarity, Epitel REMI). The Predetermined Change Control Plan (PCCP) model used by Epitel establishes precedent for continuously-improving EEG AI.

## Data Advantage

This project has immediate access to **all 7 TUH EEG corpora** - the same dataset used to train every major EEG foundation model (Neuro-GPT, EEGFormer, LaBraM). The difference: we use **all annotated sub-corpora** for multi-task learning, while existing models typically fine-tune on a single task. The combination of TUAR + TUAB + TUSZ + TUEV + TUSL + TUEP in a unified pipeline is novel.

## References

1. Obeid, I., & Picone, J. (2016). The Temple University Hospital EEG Data Corpus. *Frontiers in Neuroscience*, 10, 196.
2. Shah, V. et al. (2018). The Temple University Hospital Seizure Detection Corpus. *Frontiers in Neuroinformatics*, 12:83.
3. Cui, J. et al. (2024). Neuro-GPT: Towards a Foundation Model for EEG. *arXiv:2311.03764*.
4. Jiang, Y. et al. (2024). Large Brain Model for Learning Generic Representations (LaBraM). *ICLR 2024*.
5. Critical Care EEG Monitoring Utility. *Critical Care* (2024). doi:10.1186/s13054-024-04986-0.
