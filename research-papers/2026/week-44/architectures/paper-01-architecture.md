# Architecture: Proving and Improving the Platonic Representation Hypothesis with JEPAs (Paper 01)

Back to the analysis: [../papers/paper-01-analysis.md](../papers/paper-01-analysis.md). Raw diagram: [../diagrams/paper-01-system.mmd](../diagrams/paper-01-system.mmd).

The paper's own figures are mostly plots: the theory check (Fig. 2), connectivity panels (Fig. 3), alignment curves (Figs. 5-9, 11) and caption examples (Fig. 10). It contains no block-diagram of the system. Every diagram below is therefore a **[Conceptual reconstruction]** built from the paper's method section and theorems, unless it is marked otherwise.

## Level 1: concept [Conceptual reconstruction]

```mermaid
flowchart LR
    Z["World state z"] --> M1["Modality 1 rendering g_1(z), e.g. image"]
    Z --> M2["Modality 2 rendering g_2(z), e.g. captions"]
    Z --> M3["Modality 3 rendering g_3(z), e.g. video frames"]
    M1 --> J1["JEPA 1 trained alone"]
    M2 --> J2["JEPA 2 trained alone"]
    M3 --> J3["JEPA 3 trained alone"]
    J1 --> H["Same h*(z) up to rotation Q_m"]
    J2 --> H
    J3 --> H
    H --> USE["Fit one rotation per pair of modalities: retrieval, captioning, zero-shot"]
```

**Components and flow.**

- **World state $z$.** The scene itself. Each modality only observes a rendering of it.
- **Per-modality JEPA.** An encoder trained with a rotation-invariant prediction loss plus anti-collapse. No cross-modal data is used.
- **Theorem 1.** If every JEPA uses the *same* pairing distribution $\pi$ over world states (good views), all optima equal $\boldsymbol{Q}_m h^\star$.
- **Use.** A rotation $\boldsymbol{Q}_m\boldsymbol{Q}_{m'}^\top$, fitted by Procrustes on $\ge k$ pairs (Proposition 2), aligns any two modalities.

## Level 2: components of the experimental system [Conceptual reconstruction]

```mermaid
flowchart TD
    subgraph TXT["Text JEPA (BERT-style, 11M-335M, trained from scratch)"]
      CAPS["Captions grouped by image"] --> GV["Good-view sampler: 8 views from different captions"]
      GV --> TRIM["Trim: global 50-100%, local 15-50%; mask 15%"]
      TRIM --> ENC_T["12-layer encoder, CLS token"]
      ENC_T --> PROJ_T["Projector"]
      PROJ_T --> LOSS_T["LeJEPA: predict mean of global views + SIGReg (lambda 0.02)"]
    end
    subgraph IMG["Image JEPA (ViT-S/16) or frozen DINOv2 / DINOv3 / CLIP"]
      PIX["Image"] --> CROP["2 global 224² + 6 local 96² crops"]
      CROP --> ENC_I["ViT, CLS token"]
      ENC_I --> PROJ_I["Projector"]
      PROJ_I --> LOSS_I["LeJEPA + SIGReg"]
    end
    ENC_T --> ALIGN["Orthogonal Procrustes on paired validation set"]
    ENC_I --> ALIGN
    ALIGN --> RET["Retrieval: mutual 10-NN, R@1, CKA"]
    ALIGN --> CAPT["Captioner: text-only decoder fed rotated image embeddings"]
    ALIGN --> ZS["Pair-free alignment: Sinkhorn OT + Procrustes"]
```

**Component notes.**

- **Good-view sampler.** This is the key design choice. Positives come from *different* captions of the same image, or different frames or captions of the same video clip. The paper's ablation finds trimming is the most important knob: whole captions as views drop R@1 from .246 to .185 (Fig. 7).
- **Encoders.** The text side uses BERT-T 11M, BERT-S 33M, BERT-B 109M or BERT-L 335M. The image side is ViT-S/16, or frozen DINOv2 ViT-B/14 for most alignment tables.
- **Loss.** LeJEPA invariance to the mean of the two global views, with SIGReg anti-collapse. AdamW, batch 256, 200 epochs.
- **Alignment.** One orthogonal map. When encoder widths differ (768 vs 384), it is a map with orthonormal rows.
- **Captioner.** It is trained only on text, by reconstructing captions from noisy text embeddings with $\sigma\approx3$. At test time it receives image embeddings rotated into text space.

## Level 3: low-level execution and data flow [Conceptual reconstruction]

```mermaid
flowchart TD
    B0["Batch: B images, each with several captions"] --> V1["Text views: B x 8 x L tokens (L ≤ 64)"]
    V1 --> E1["Encoder: B x 8 x 384 CLS (768 for BERT-B)"]
    E1 --> P1["Projector: B x 8 x k'"]
    P1 --> C1["Centre: mean of 2 global views, B x 1 x k'"]
    P1 --> PRED["Prediction loss: mean over views of squared distance to centre"]
    C1 --> PRED
    P1 --> FLAT["Flatten to (B*8) x k'"]
    FLAT --> RP["Random unit directions a_s, project to (B*8) x S"]
    RP --> EP["Epps-Pulley statistic vs N(0,1) per direction"]
    EP --> SIG["SIGReg = mean over directions"]
    PRED --> TOT["Total = prediction + 0.02 * SIGReg"]
    SIG --> TOT
    TOT --> OPT["AdamW step; clip gradients on long runs"]
    OPT -.->|"after training"| FRZ["Frozen CLS embeddings for n paired items"]
    FRZ --> SVD["SVD of A^T B = U S V^T"]
    SVD --> ROT["R = U V^T"]
    ROT --> EVAL["Rotated text vs image embeddings: kNN / retrieval / decoding"]
```

**Data-flow notes.**

- The prediction term and SIGReg act on projector outputs. The visible text does not specify the projector width, so it is shown as $k'$.
- SIGReg is computed per GPU batch and scales with batch size. The paper flags this as a reason the effective weight changes between hardware setups.
- Procrustes is solved in closed form once. With zero pairs, the paper alternates Sinkhorn optimal transport (to guess correspondences) with Procrustes.

## Theory-to-architecture map [Conceptual reconstruction]

```mermaid
flowchart LR
    T1["Theorem 1: same pi gives same h* up to rotation"] --> D1["Design: use good views in every modality"]
    P1["Prop. 1: spectral optimum = eigenvectors of K_pi"] --> D2["Design: whitening / SIGReg anti-collapse"]
    T2["Theorem 2: misalignment ≤ kernel gap / eigengap"] --> D3["Expectation: mismatched views degrade gracefully"]
    T3["Theorem 3 + Prop. 3: nuisance costs alignment"] --> D4["Design: views from different captions remove wording nuisance"]
    P2["Prop. 2: k pairs identify the rotation"] --> D5["Design: Procrustes on a few pairs"]
    P4["Prop. 4: NTP is a bad-view JEPA"] --> D6["Explains why LLMs align only through scale"]
```
