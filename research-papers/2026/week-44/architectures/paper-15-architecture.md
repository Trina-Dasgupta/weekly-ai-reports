# Paper 15 — Architecture: ThinkWorld

Back to the analysis: [../papers/paper-15-analysis.md](../papers/paper-15-analysis.md)

Provenance: the paper's Fig. 1 gives the concept overview, and Fig. 2 gives the architecture with (a) the SFT stage and (b) the RL stage. Diagrams labelled **[From paper: Fig. X — redrawn]** follow those figures' described structure, with component sizes from Section 3 and Appendix A. The others are **[Conceptual reconstruction]**.

---

## Level 1 — Concept **[From paper: Fig. 1 — redrawn as a flow]**

```mermaid
flowchart LR
    R["Robot demos (with actions)"] --> COD["Chain-of-Dynamics: discrete motion thoughts"]
    HV["Human videos (no robot actions)"] --> COD
    COD --> IM["World imagination: predicted future features"]
    COD --> ACT["Robot actions"]
    IM --> RW["Compare with observed future -> reward"]
    RW -. "reinforce" .-> COD
```

**Explanation.** Both data sources meet in one motion-token reasoning space. Imagination makes every video a source of reward, and actions are decoded from the same grounded context.

---

## Level 2 — Components **[From paper: Fig. 2 — redrawn]**

```mermaid
flowchart TD
    IN["Instruction l, history h_t (0-16 frames, 2 Hz), current view o_t, wrist views, state (256 bins)"] --> VLM["Qwen3-VL-2B-Instruct"]
    VLM --> COD["CoD tokens (FAST motion vocab 2,048; 1-s wrist deltas, 6D or 12D)"]
    COD --> Q64["64 learned future-query tokens"]
    Q64 --> PJ["Future projector: 2-layer MLP + 6-layer cross-attention decoder"]
    PJ --> Z["Predicted DINO features 196 x 1280"]
    F1["Main-view frame at t + 1 s"] --> DV3["Frozen DINOv3 ViT-H/16+ (224 x 224, target only)"]
    DV3 --> ZS["Target features 196 x 1280"]
    VLM -- "layer-wise K/V of input tokens only" --> AX["Action expert (from MolmoAct2): 36 non-causal blocks, width 768, FP32"]
    AX --> ACH["Action chunk (max 30 x 32, masked per embodiment)"]
```

| Component | Role | Trained in |
|---|---|---|
| Qwen3-VL-2B VLM | Encodes context, generates CoD, hosts future queries | Stage 1 SFT, Stage 3 RL |
| Motion tokenizer | Turns robot/human wrist motion into CoD targets | Fixed |
| Future projector | Maps future queries to DINO-space predictions | Stage 1; frozen in RL |
| DINOv3 | Produces target features of the real future | Always frozen |
| Action expert | Flow-matching decoder of continuous actions | Stage 2 (VLM frozen); frozen in RL |

### Training-stage view **[From paper: Fig. 2(a)/(b) — redrawn; stage details from Section 3]**

```mermaid
flowchart LR
    S1["Stage 1 VLM SFT: L_CoD + lambda L_future (150k it, 128 H100)"] --> S2["Stage 2 action SFT: masked flow matching, VLM frozen (50k it)"]
    S2 --> S3["Stage 3 RL: GRPO on r_world, K=4, VLM only (2k it, 64 H100)"]
    S3 --> S4["Embodiment post-training: LIBERO or YAM"]
```

---

## Level 3 — Low-level execution

### 3a. Inference step **[Conceptual reconstruction]**

```mermaid
sequenceDiagram
    participant Cam as Cameras (top + 2 wrist)
    participant V as Qwen3-VL-2B
    participant E as Action expert
    participant Rb as Robot (YAM)
    Cam->>V: history frames + current images + instruction + state
    V->>V: prefill input tokens (cache layer-wise K/V)
    V->>V: decode CoD motion tokens
    V->>E: layer-wise K/V of input tokens
    E->>E: 10 Euler steps of flow field from noise
    E->>Rb: action chunk (e.g. 30 x 14 absolute joints)
    Note over V,E: Table 9 reports p50 324.84 ms, 7.07 GiB on H100, batch 1
```

### 3b. RL reward computation **[Conceptual reconstruction]** (from Eq. 4, Eq. 6 and the GRPO settings)

```mermaid
flowchart TD
    C["Context"] --> S["Sample 4 CoDs"]
    S --> P["Frozen projector -> z^(k), 196 x 1280 each"]
    FT["Observed frame t+1s"] --> D["DINOv3 -> z*"]
    P --> CS["Per-patch cosine s_p"]
    D --> CS
    D --> MW["Motion weights w_p: 0.2 at/below median change, linear to 1 at 90th pct"]
    CS --> R["r = 0.5 [mean s_p + weighted mean s_p]"]
    MW --> R
    R --> MF["Malformed CoD -> group min - 0.05"]
    MF --> N["Standardise within group of 4, then batch"]
    N --> G["Clipped token ratios [0.8, 1.28], KL 0.04"]
    G --> L["L_RL = 0.999 L_GRPO + 0.001 L_replay"]
    L --> U["Update VLM only"]
```

**Data-flow note.** In human-video batches, the action-expert path is absent: no action loss is computed, and only CoD and future-prediction or reward signals train the VLM.
