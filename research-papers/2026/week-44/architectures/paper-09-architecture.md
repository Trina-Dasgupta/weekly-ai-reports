# Paper 09 architecture — Nemotron-Labs-Diffusion-Uni (NLD-Uni)

Back to the analysis: [../papers/paper-09-analysis.md](../papers/paper-09-analysis.md) · Raw diagram: [../diagrams/paper-09-system.mmd](../diagrams/paper-09-system.mmd) · Paper: arXiv:2610.36466v1 (NVIDIA-branded report; per-author affiliations not listed in the PDF) · Code: https://github.com/NVlabs/Nemotron-Labs-Diffusion-Uni · Weights: https://huggingface.co/nvidia/NLD-Uni

NLD-Uni is a unified **discrete diffusion LLM** for image understanding and image generation/editing, built as a two-tower Elastic Mixture-of-Transformers, trained with **cross-paradigm on-policy distillation** and a **self-refining** corruption process for image tokens.

---

## Level 1 — Concept

**[From paper: Fig. 1 (left)]**, redrawn

```mermaid
flowchart LR
    T1["AR VLM teacher<br/>(Nemotron-3 Nano Omni)"] -->|"understanding OPD"| S["NLD-Uni<br/>unified diffusion LLM"]
    T2["Rectified-flow teacher<br/>(Qwen-Image)"] -->|"editing OPD"| S
    S --> U["Understanding: VQA, math reasoning, object/GUI grounding"]
    S --> G["Generation: text-to-image, image editing"]
```

**Idea.** One sequence model, one diffusion objective over text and image tokens; borrow skills from best-in-class teachers of *other* paradigms.

---

## Level 2 — Components

**[From paper: Figs. 2–3]**, redrawn

```mermaid
flowchart TD
    IMG["Input image"] --> SEM["Semantic encoder (RADIO-H)<br/>continuous embeddings"]
    IMG --> VQ["VQ encoder (IBQ, 131k codes)"]
    TXT["Text instruction"] --> TT["Text tokens"]
    SEM --> SEQ["Block sequence: clean prompt blocks + corrupted output block(s)"]
    VQ --> SEQ
    TT --> SEQ
    SEQ --> UT["Understanding tower<br/>(init. NLD diffusion LLM)"]
    SEQ --> GT["Generation tower<br/>(init. copy of understanding)"]
    UT <-->|"joint attention"| GT
    UT --> OT["Clean text prediction"]
    GT --> OI["Clean VQ-code prediction"]
    OI --> DEC["VQ decoder -> image"]
```

| Component | Detail (as reported) |
|---|---|
| Backbone formulation | Set Block Decoding (position-shifted variant); block size randomised in training; each image is one block of resolution-dependent size |
| Understanding tower | Initialised from Nemotron-Labs-Diffusion (NLD) LLM |
| Generation tower | Initialised as a copy of the understanding tower |
| Elastic views (Fig. 2) | VLM view 8B active; T2I view 12B (4B understanding + 8B generation); interleaved view 16B |
| Input image encoding | RADIO-H semantic encoder + IBQ VQ encoder |
| Output image encoding | VQ codes only |
| Text corruption | standard binary masking |
| Image corruption | self-refining ball-tree process (below) |

---

## Level 3 — Low-level mechanics

### 3a. Self-refining image-token corruption

**[From paper: Fig. 4 and Fig. 11]**, redrawn

```mermaid
flowchart LR
    C0["age 0: clean code c<br/>leaf of ball tree (depth D = 18)"] --> C1["age 1: random code in N_17(c)<br/>2 codes"]
    C1 --> C2["age 2: random code in N_16(c)<br/>4 codes"]
    C2 --> CD["... ball doubles each age ..."]
    CD --> CK["age K = 17: uniform over vocabulary<br/>(acts as fully masked)"]
```

- Each token \(i\) gets a window \((\tau_i,\tau_i+\epsilon)\); age rises linearly through the window (Eq. 16). \(\epsilon\to 0\) recovers ordinary one-jump masked diffusion.
- Reverse step: predict clean code \(\hat c\); re-sample from \(N_{D-a}(\hat c)\) at the new, lower age (Eq. 12). Tokens are committed coarsely, then refined — and may move to a different ball if the prediction changes.
- Tree stored as a 1-D array (524 KB) instead of a 131k×131k distance matrix (32 GB).
- Default inference: 64 NFE with 8 refinements per token (Table 6).

### 3b. Block-diffusion training stream

**[Conceptual reconstruction]** from Appendix A.1 and Fig. 10

```mermaid
flowchart LR
    X0["Clean blocks B0_1 .. B0_K"] --> CAT["Concatenate: [clean stream | noisy stream]"]
    XT["Noisy blocks Bt_1 .. Bt_K (forward corruption)"] --> CAT
    CAT --> ATT["Attention mask:<br/>clean = causal;<br/>noisy block = bidirectional within block + all earlier clean blocks"]
    ATT --> LAR["AR cross-entropy on clean stream (AR view)"]
    ATT --> LDF["Diffusion cross-entropy (1/t weighted) on noisy stream"]
```

### 3c. Understanding OPD (transitive distillation)

**[From paper: Fig. 5]**, redrawn

```mermaid
sequenceDiagram
    participant S as Student (AR view, block size 1)
    participant T as Frozen AR teacher
    participant D as Student (diffusion view)
    S->>S: roll out block B0_i at low temperature
    S->>T: score rollout
    T-->>S: AR OPD loss = KL(student AR || teacher AR)
    S->>D: corrupt B0_i -> Bt_i
    D-->>S: consistency loss = KL(stop-grad student AR || student diffusion) on masked positions
```

### 3d. Generation OPD (teacher-refined student output)

**[From paper: Fig. 6 and Algorithm 1]**, redrawn

```mermaid
flowchart LR
    R["Student editing rollout (VQ codes)"] --> P["VQ decode -> pixels"]
    P --> E["Teacher VAE encode: x_0"]
    E --> N["Noise: x_r = (1 - r) x_0 + r * delta, r = 0.9"]
    N --> O["Teacher ODE solve (multi-step) -> x_hat_0"]
    O --> Q["Decode + VQ re-encode (stop-grad) -> target X_hat_0"]
    Q --> L["SFT loss on corrupted target: -(1/t) log p(X_hat_0 given X_hat_t)"]
```

### Training stages

**[Conceptual reconstruction]** from Sec. 4.1, Appendix B and Table 5

```mermaid
flowchart LR
    S1["Stage 1<br/>text-image pairs (127M images)<br/>400k steps, lr 1e-5"] --> S2["Stage 2<br/>+ instruction, grounding, editing data<br/>100k x 2 steps, lr 1e-5<br/>-> NLD-Uni-Base"]
    S2 --> S3["Stage 3<br/>SFT + OPD (76.8k reasoning, 100k editing prompts)<br/>10k steps, lr 2e-6 -> NLD-Uni"]
```

Compute: 64 H100 GPUs (8 nodes), 26 days total, OPD stage 5 days (Appendix B.3).
