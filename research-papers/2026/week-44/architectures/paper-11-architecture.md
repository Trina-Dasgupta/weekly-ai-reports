# Paper 11 architecture — Rollback (experience backtracking in world models)

Back to analysis: [../papers/paper-11-analysis.md](../papers/paper-11-analysis.md) · Raw diagram: [../diagrams/paper-11-system.mmd](../diagrams/paper-11-system.mmd)

Rollback has two layers: an **algorithm** (tree search in a rewindable world model → relative value labels → advantage-conditioned policy; real data → world-model adaptation) and a **system** (cloud–edge–device asynchronous training).

## Level 1 — Concept **[From paper: Fig. 1c and Fig. 4]** (redrawn)

```mermaid
flowchart LR
    ROBOT["Real robot executes latest policy"] -->|"real trajectories D_real"| WMA["World-model adaptation (LoRA SFT + DMD)"]
    ROBOT -->|"seeds + direct supervision"| SEARCH["Rollback search in world model"]
    WMA -->|"updated world model"| SEARCH
    SEARCH -->|"tree labels D_tree"| PT["Policy training"]
    PT -->|"new policy snapshot"| ROBOT
    PT -->|"new proposal policy"| SEARCH
```

- Forward-rollout world-model RL (Fig. 1b) wastes budget on independent rollouts. Rollback (Fig. 1c) rewinds to key decisions and learns from outcome differences, while real feedback refines the world model.

## Level 2 — Components **[From paper: Fig. 2 and Fig. 3]** (redrawn and merged)

```mermaid
flowchart TD
    subgraph CLOUD["Cloud"]
        ORCH["Elastic Orchestrator (queue-aware, state-affine placement)"]
        subgraph POOL["Distributed Search Pool: GPU workers x N"]
            POL1["Policy replica"]
            WM1["World model GE-Sim 2.0"]
            CTX["Context cache"]
            JUDGE["World judge (success classifier)"]
        end
        FAB["Experience Fabric: real stream + tree stream, sharded replay, source-aware sampling"]
        PTR["Policy Trainer (DDP)"]
        WTR["WM Trainer (LoRA SFT then DMD)"]
        PUB["Publish: versioned snapshots"]
    end
    subgraph EDGE["Edge site"]
        RT["Policy Runtime + weight cache"]
        BUF["Trajectory buffer"]
    end
    subgraph DEV["Device"]
        CTRL["Robot controller 30 Hz"]
    end
    ORCH -->|"seed trajectories"| POOL
    POOL -->|"finished tree supervision"| FAB
    FAB -->|"D_mix batches"| PTR
    FAB -->|"real trajectories only"| WTR
    PTR --> PUB
    WTR --> PUB
    PUB -->|"policy + WM snapshots"| POOL
    PUB -->|"policy snapshots"| RT
    CTRL -->|"observations"| RT
    RT -->|"action chunks"| CTRL
    RT --> BUF
    BUF -->|"labelled episodes"| FAB
```

| Component | Role | Key details (paper) |
|---|---|---|
| GE-Sim 2.0 world model | Predict 3-view frames + proprio state from context + action chunk | 25 frames per chunk at 16 fps; pose-expert head for proprio (Sec. 4.1, Table 5) |
| Context cache | Store `m_t` per node so branches restore instead of regenerate | Kept in GPU memory on the worker for the whole tree (Sec. 5.2) |
| World judge | Binary success on imagined rollouts | Fine-tuned with BCE on annotated task videos; threshold 0.5 |
| Policy (π0.5) | Proposes chunks (highest bin) and is trained | 50-step chunks, 16-D actions, K=5 bins, p_drop 0.5, CFG β=1.5 |
| Orchestrator | Assign seeds to idle workers, recover workers | Workers don't communicate during search |
| Experience Fabric | Only data exchange point | Separate real / tree streams; prefetch to learners |
| Policy Trainer | Optimise Eq. 5 continuously | 8 H200, DDP |
| WM Trainer | Test-time physics grounding | 400 LoRA SFT + 100 DMD steps per adaptation |
| Edge runtime | Inference next to robot | Swaps weights only at episode boundaries |

## Level 3 — Search and label extraction data flow **[Conceptual reconstruction]** from Algorithms 2–3, Eq. 2–5

```mermaid
flowchart TD
    S1["Seed tau with outcome y_tau"] --> S2["Spine nodes at chunk boundaries, back up y_tau"]
    S2 --> S3["Depth limit: spine depth if success else D_max"]
    S3 --> SEL["Select v: argmax Q(u) + c_puct P(u given v) sqrt(1+N(v))/(1+N(u)), widening cap c_pw N^alpha"]
    SEL --> RES["Restore cached m_v"]
    RES --> SAMP["Sample chunk a (50 x 16) from policy with bin K, fresh noise"]
    SAMP --> STEP["World model step: frames + proprio, cache m at child"]
    STEP --> ROLL["Roll out highest-bin policy to success or depth limit"]
    ROLL --> JUD["Judge leaf: y in {0,1}"]
    JUD --> BACK["Back up: N += 1, W += y, Q = W/N"]
    BACK --> MIX{"Siblings in different provisional bins?"}
    MIX -- "yes" --> REF["Refine least-visited sibling until N >= n_conf"]
    MIX -- "no" --> BUD{"n_sim < B = 256?"}
    REF --> BUD
    BUD -- "yes" --> SEL
    BUD -- "no" --> LAB["Labels: q_hat = mean leaf y, rho = q_hat/q_max, b = min(K, 1 + floor(K rho))"]
    LAB --> DROP{"All siblings same bin or q_max = 0?"}
    DROP -- "yes" --> DISC["Discard group"]
    DROP -- "no" --> OUT["Add (o_v, g, a_i, b_i) to D_tree"]
```

**Data shapes**
- Observation: 3 RGB views on a 384×512 canvas + proprio (2 × (7 joints + 1 gripper) = 16 values).
- Action chunk: `50 × 16`.
- Tree sample: `(o_v, g, a_i, b_i)` with `b_i ∈ {1..5}`.
- Flow-matching batch: noisy chunk `x_s = (1−s)a + sε`, target velocity `ε − a` (both `50×16` per sample).

## Version pinning and asynchrony **[From paper: Fig. 4, Sec. 5.2–5.3]** (redrawn)

```mermaid
sequenceDiagram
    participant R as Robot (edge)
    participant F as Experience Fabric
    participant S as Search worker
    participant P as Policy trainer
    participant W as WM trainer
    S->>S: pin latest policy v_i and WM w_j for this tree
    S->>F: tree supervision when tree finishes
    P->>F: fetch D_mix batch
    P-->>S: publish policy v_i+1 (picked up at next tree)
    P-->>R: publish policy v_i+1 (switch at episode boundary)
    R->>F: labelled real episode
    W->>F: fetch D_real
    W-->>S: publish WM w_j+1 (next tree)
```

No process waits for another; they exchange only data and versioned snapshots (Sec. 5).
