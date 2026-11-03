# Paper 06 architecture — EmbeddingGemma 2

Back to the analysis: [../papers/paper-06-analysis.md](../papers/paper-06-analysis.md) · Raw diagram: [../diagrams/paper-06-system.mmd](../diagrams/paper-06-system.mmd)

EmbeddingGemma 2 = a 271M bidirectional Gemma 4 backbone (24 layers, width 512, table-free per-layer embeddings) + a 167M trainable ViT + a 305M frozen Conformer audio encoder, mean-pooled into a 768-d Matryoshka embedding. All 744M parameters together; modular subsets of 271M / 439M / 577M.

---

## Level 1 — concept  **[Conceptual reconstruction from Sec. 2]**

```mermaid
flowchart LR
    ANY["Any input: text, image, audio, video, or interleaved mix"] --> ENC["Modality encoders to soft tokens"]
    ENC --> BB["One bidirectional backbone"]
    BB --> VEC["One normalized vector in a shared space"]
    VEC --> USE["Search / RAG / clustering / classification"]
```

**Explanation.** Unlike CLIP-style two-tower models, every modality flows through the **same** backbone, so a query mixing a photo and a question gets a single vector comparable to text, images, audio or video documents.

---

## Level 2 — components  **[Conceptual reconstruction from Sec. 2 (architecture and input handling)]**

```mermaid
flowchart TB
    subgraph IN["Input construction T"]
        TP["Task prefix + text: embedding lookup"]
        VE["ViT, patch 16, 167M, trainable: N_vis soft tokens (70-1120), 768 to 512"]
        AE["2 conv + 12 Conformer (USM), 305M, frozen: 25 tok/s, 1536 to 512"]
        VID["Video: 1 fps, max 32 frames x 140 tokens, optional 1-s audio + MM:SS"]
    end
    TP --> SEQ["BOS + segments + EOS (max 8192)"]
    VE --> SEQ
    AE --> SEQ
    VID --> SEQ
    SEQ --> BACK["24 bidirectional layers, d_M = 512, table-free PLE per layer"]
    BACK --> POOL["Mean pool P"]
    POOL --> PROJ["Linear f: 512 to 768"]
    PROJ --> NORM["L2 normalize"]
    NORM --> MRL["Matryoshka views: 768 / 512 / 256 / 128"]
```

| Component | Params | Trainable | Notes |
|---|---|---|---|
| Backbone (incl. embedder) | 271M | Yes | Gemma 4 decoder weights, causal mask flipped to bidirectional; n = 24, d_M = 512 |
| Vision encoder | 167M | Yes (multimodal stages) | Init from smaller Gemma 4 vision encoder; trainable gives +4.09 MMEB-v2 (Table 4) |
| Audio encoder | 305M | Frozen | Bit-identical to Gemma 4 E2B/E4B audio encoder |
| Pooling + projection | small | Yes | Mean pool → linear (random init) → ℓ2 |
| Quantized deployment | — | QAT tail | Backbone int4 (Q4_0 / Q4_C), vision W8A8, audio mixed 2/4/8-bit weights |

---

## Level 3 — inside one backbone layer (table-free PLE)  **[Conceptual reconstruction from the PLE equations]**

```mermaid
flowchart LR
    X["H_(l-1)"] --> ATT["Bidirectional self-attention + FFN"]
    ATT --> HP["H'_l"]
    TIN["T_in (shared input embeddings)"] --> P["p_l, RMSNorm: P_l"]
    HP --> G["g_l, GELU: G_l"]
    G --> M["G_l * P_l"]
    P --> M
    M --> U["u_l, RMSNorm: U_l"]
    HP --> S["H_l = H'_l + U_l"]
    U --> S
    S --> NEXT["to layer l+1"]
```

**Data flow.** Each layer re-reads the original token embeddings through its own projection $p_l$, gates them with a GELU of its current hidden state, projects back and adds to the residual stream. Gemma 3n/4 used a separate large PLE lookup table for this; EmbeddingGemma 2 removes it and derives the per-layer signal from the shared input embeddings, scoring slightly higher (MMEB-v2 55.46 vs 55.07, Table 2).

---

## Training flow  **[Conceptual reconstruction from Sec. 3]**

```mermaid
sequenceDiagram
    participant G4 as Gemma 4 decoder init
    participant TPF as Text pre-finetune (~300B)
    participant MPF as Multimodal pre-finetune (~2T)
    participant MFT as Multimodal finetune (~600B)
    participant MG as Merge
    participant QAT as QAT tail
    G4->>TPF: flip mask to bidirectional
    TPF->>MPF: distillation from Gemini Embedding 2 (+ small contrastive weight)
    MPF->>MFT: distillation only; image / audio / video / docs; larger batch and LR
    MFT->>MG: contrastive NCE + spread-out + MRL; smaller LR and batch
    MG->>QAT: average final with much earlier checkpoint
    QAT-->>QAT: distil quantized student from frozen float model
```
