# Paper 05 architecture — Hy-Nomas

Back to the analysis: [../papers/paper-05-analysis.md](../papers/paper-05-analysis.md) · Raw diagram: [../diagrams/paper-05-system.mmd](../diagrams/paper-05-system.mmd)

Hy-Nomas is an 8B causal video world model (HunyuanVideo 1.5 lineage) that generates video segment by segment, steered by free-form text and continuous camera control, with a Prompt Enhancer in front and a KV cache of history behind.

---

## Level 1 — concept  **[From paper: Fig. 1 — redrawn]**

```mermaid
flowchart LR
    USER["User: text command or WASD / mouse"] --> PE["Prompt Enhancer"]
    PE -->|"structured caption"| WM["Hy-Nomas world model"]
    USER -->|"camera trajectory"| WM
    MEM["History (KV cache)"] --> WM
    WM --> OUT["Next video segment"]
    OUT --> MEM
    OUT -->|"seen by PE next turn"| PE
```

**Explanation.** Each turn is one segment. The PE converts messy user intent into a strict JSON caption (scene changes + timestamped actions). The world model renders the segment conditioned on that caption, an optional camera path, and cached history; the result is cached for the next turn.

---

## Level 2 — components  **[From paper: Fig. 5, Table 1 — redrawn]**

```mermaid
flowchart TB
    subgraph COND["Conditioning"]
        CAP["Structured JSON caption: head + timestamped action tails"] --> TE["Frozen Qwen2.5-VL-7B text encoder"]
        TE --> REF["2-layer token refiner, 3584 to 2048"]
        POSE["Per-frame camera poses"] --> REL["Relative pose, scale by 99th percentile"]
        REL --> GRP["Group 4 poses = 64-D, linear + residual MLP"]
    end
    subgraph DIT["Causal MMDiT backbone (8B)"]
        BLK["54 dual-stream MMDiT blocks, d = 2048, 16 x 128 heads, 3D RoPE (16,56,56)"]
        CX["Camera cross-attention per block (width 1024, 16 heads)"]
        MASK["Action-aware causal mask"]
    end
    subgraph LAT["Latent space"]
        ENC["DCAE-V: 32x spatial, 4x temporal, 32 ch"]
        DEC["One-step PixelUMM-based pixel decoder (DMD2, FP8)"]
    end
    REF --> BLK
    GRP --> CX
    CX --> BLK
    MASK --> BLK
    ENC -->|"history latents"| KV["INT8 KV cache, sliding window"]
    KV --> BLK
    BLK -->|"velocity, 4 NFE"| DEC
```

| Component | Role | Key numbers |
|---|---|---|
| DCAE-V autoencoder | Video ↔ latent | 32× spatial, 4× temporal, 32 channels, patch 1×1×1 |
| Text encoder + refiner | Caption → tokens | Qwen2.5-VL-7B frozen, width 3584, max 1280 tokens → 2048 |
| MMDiT backbone | Joint text/video denoiser | 54 blocks, width 2048, MLP 8192, 16 heads × 128 |
| Camera branch | Continuous 6-DoF control | 64-D grouped extrinsics, cross-attn width 1024, one per block |
| Action-aware mask | Temporal grounding of actions | Heads persist; tails only in own window and segment |
| Prompt Enhancer | Multi-turn intent → structured caption | SFT VLM (named Qwen3.8-27B); ~100% format adherence (text) |
| Inference stack | Consumer GPU | INT8 weights/KV, sparse attn (25% blocks) after 2 dense steps, 20.53 GiB peak on RTX 4090 for 100 s @ 1280×704, 24 FPS |

---

## Level 3 — low-level execution of one turn  **[Conceptual reconstruction from Sec. 3–6]**

```mermaid
sequenceDiagram
    participant U as User
    participant PE as Prompt Enhancer
    participant TE as Text encoder + refiner
    participant D as Causal MMDiT (4 NFE)
    participant KV as INT8 KV cache
    participant DEC as Pixel decoder
    U->>PE: turn t input u_t (text or camera)
    PE->>PE: y_t ~ p(. | S, u_t, H_(t-1), V_(<t), m_t)
    PE->>TE: description_change + contained_actions (text turn) or description_change only (camera turn)
    TE->>D: text tokens (B x L_txt x 2048)
    U->>D: camera tokens, one per latent frame (camera turn)
    KV->>D: cached K/V of earlier heads + video (tails of earlier segments excluded)
    loop 4 denoising steps (dense attention first 2, then sparse 25%)
        D->>D: predict 32-ch latent velocity, Euler update
    end
    D->>DEC: clean latent (B x 32 x F_z x H/32 x W/32)
    DEC->>U: RGB frames of segment t
    D->>KV: re-noise latent at sigma = 0.206897, append K/V (evict beyond window)
```

**Training-time counterpart (Fig. 7):** segment forcing replaces ground-truth history with the model's own one-step low-noise estimate ($t_{\mathrm{ctx}}\sim\mathcal{U}(0.1,0.3)$), applies the flow-matching loss per segment, and backpropagates each segment immediately (streaming backward) before accumulating one optimizer step.
