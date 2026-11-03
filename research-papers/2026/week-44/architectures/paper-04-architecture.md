# Paper 04 architecture — PDRVLA

Back to the analysis: [../papers/paper-04-analysis.md](../papers/paper-04-analysis.md) · Raw diagram: [../diagrams/paper-04-system.mmd](../diagrams/paper-04-system.mmd)

PDRVLA = a Qwen3.5-4B VLM + a 57.3M-parameter **Physical Dynamics Module (PDM)** + a ~150M DiT **action expert** trained with flow matching. A frozen video world model is used only during pretraining to shape the PDM's 50 "physical tokens".

---

## Level 1 — concept  **[From paper: Fig. 1a / Fig. 3a — redrawn]**

```mermaid
flowchart LR
    IN["Images + instruction + proprio state"] --> VLM["VLM (Qwen3.5-4B)"]
    VLM -->|"mid-layer features"| PDM["Physical Dynamics Module"]
    VLM -->|"final-layer features"| AE["Action expert"]
    PDM -->|"physical tokens"| AE
    AE --> OUT["8-step action chunk"]
    WM["Frozen video world model (training only)"] -.->|"supervises"| PDM
```

**Explanation.** The VLM understands *what* is in the scene and *what* the task is. The PDM summarizes *how the scene is about to change*. The action expert combines both to generate actions. The dotted edge is the key idea: the world model teaches the PDM during training but is absent at inference.

---

## Level 2 — components and training stages  **[From paper: Fig. 3b, Fig. 4 — redrawn]**

```mermaid
flowchart TB
    subgraph VLMB["VLM backbone: 32 decoder blocks"]
        L8["Block 8 output H8"]
        LL["Final block output H_last"]
    end
    subgraph PDMB["Physical Dynamics Module (57.3M)"]
        Q["50 learnable queries (width 1024)"]
        B1["Block x4: cross-attn to H8, self-attn, FFN ratio 1, 4 heads"]
        PJ["Projection to 50 x 2560"]
        Q --> B1 --> PJ
    end
    L8 --> B1
    subgraph AEB["Action expert (DiT ~150M)"]
        CND["Condition X_t = concat(H_last, C_t) + state prefix"]
        FM["Flow-matching velocity net"]
        CND --> FM
    end
    LL --> CND
    PJ --> CND
    subgraph TRN["Training-only supervision"]
        DH["Direct head: LN + linear, spatial-query pooling, RGB/depth/flow readouts"]
        WM["Frozen LingBot-Video-Dense-1.3B + frozen Wan VAE"]
        DF["Trainable depth / flow heads on WM hidden states"]
    end
    PJ --> DH
    PJ -->|"stage 1 only"| WM
    WM --> DF
```

| Component | Role | Trainable in stage 1 | Trainable in stage 2 |
|---|---|---|---|
| VLM (Qwen3.5-4B) | Perception + language; taps at block 8 and final block | No | Yes (final-layer path + VL loss) |
| PDM | Produces physical tokens C_t | Yes | Yes (input detached) |
| Direct head | Regresses standardized future RGB/depth/flow latents | Yes | Yes (weight 0.1) |
| Frozen video WM + depth/flow heads | Denoises future latents conditioned on C_t | WM frozen, heads trainable | Removed |
| Action expert (DiT) | Flow-matching action generator, K = 8, 58-dim | — | Yes |

**Data flow in training.** Stage 1 loss: $\mathcal{L}_{\mathrm{pre}}=\mathcal{L}_{\mathrm{WM}}+\lambda(k)\mathcal{L}_{\mathrm{direct}}$ with λ ramped 0 → 0.5 over 2,000 updates. Stage 2 loss: $\mathcal{L}_{\mathrm{act}}+0.1\mathcal{L}_{\mathrm{VL}}+0.1\mathcal{L}_{\mathrm{direct}}$. Stage 3: per-benchmark adaptation.

---

## Level 3 — low-level execution  **[Conceptual reconstruction from Sec. 4, App. A.1]**

```mermaid
sequenceDiagram
    participant Env as Robot / simulator
    participant VLM as Qwen3.5-4B
    participant PDM as Physical Dynamics Module
    participant DiT as Action expert
    Env->>VLM: up to 4 views + "embodiment ... control Hz" + instruction (+ "Current step: subtask")
    VLM->>VLM: 32 blocks over N tokens
    VLM->>PDM: H8 (B x N x d_vlm), detached
    PDM->>PDM: 4 x [cross-attn(queries, H8), self-attn, FFN]
    PDM->>DiT: C_t (B x 50 x 2560)
    VLM->>DiT: H_last (B x N x d_vlm)
    Env->>DiT: proprio state prefix token
    DiT->>DiT: x0 = noise (B x 8 x 58)
    loop 4 Euler steps, dt = 0.25
        DiT->>DiT: x = x + 0.25 * v_hat(x, tau | X_t, state)
    end
    DiT->>Env: action chunk: per step [7 joints, 6 EEF (dp, Log dR) rel. to chunk start, 12 hand, 1 gripper] x 2 arms, masked dims ignored
```

**Execution notes**
- The PDM's cost is linear in VLM sequence length (50 queries × N keys) — tiny compared to the VLM.
- EEF actions in a chunk are all relative to the **chunk-start** pose; joint, hand and gripper values are absolute targets.
- Training-time future targets: 5-frame clips spanning the current step and the next 30 action steps, encoded by the frozen Wan VAE; depth and flow are rendered as images (grayscale / color-coded) before encoding.
- Inference latency and control frequency are not reported in the paper.
