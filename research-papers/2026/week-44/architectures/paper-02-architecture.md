# Architecture: Multi-Token Prediction Done Right, Soft-MTP (Paper 02)

Back to the analysis: [../papers/paper-02-analysis.md](../papers/paper-02-analysis.md). Raw diagram: [../diagrams/paper-02-system.mmd](../diagrams/paper-02-system.mmd).

Sources for the diagrams:

- Fig. 1 (illustration, pseudocode, loss curves)
- Fig. 2 (position-aligned NTP distributions)
- Table 1 (gradient routing)
- Algorithm 1 / Appendix B.4 (the trick)
- Appendix D (model configs)

The paper does not spell out the internal sub-layers of the DeepSeek-style head. Where a diagram shows them, they are marked as reconstruction.

## Level 1: concept [Conceptual reconstruction, informed by Fig. 1]

```mermaid
flowchart LR
    IN["Token sequence"] --> TRUNK["Shared transformer trunk"]
    TRUNK --> NTP["Next-token head (teacher at inference and in Soft-MTP)"]
    TRUNK --> HEADS["MTP heads 1..K (students / drafters)"]
    HEADS -->|"Soft-MTP: learn to match own NTP at aligned position"| NTP
    NTP --> VERIFY["Inference: verify drafts"]
    HEADS --> DRAFT["Inference: draft K tokens"]
    DRAFT --> VERIFY
```

**Idea.** The MTP heads exist to draft tokens the NTP head will accept. Soft-MTP therefore trains them, through the trunk as well, to match the NTP distribution at the aligned future position (Fig. 2) rather than the noisy one-hot data token. It does this without computing any explicit KL.

## Level 2: components and gradient routing [From paper: Table 1 and Appendix D; layout is a reconstruction]

```mermaid
flowchart TD
    EMB["Token embedding (V = 50,304, untied)"] --> TR["Trunk: 36 layers, d in {512, 768, 1152, 1536}, head dim 64"]
    TR --> H0["h0 (NTP hidden)"]
    H0 --> HD1["MTP head 1: takes h0 + embedding of shifted token, own RMSNorm"]
    HD1 --> HD2["MTP head 2 (sequential)"]
    HD2 --> HD3["MTP head 3 (sequential)"]
    H0 --> W["Shared output head W (V x d)"]
    HD1 --> W
    HD2 --> W
    HD3 --> W
    W --> L0["NTP cross-entropy (fused)"]
    W --> LK["MTP cross-entropy x3 (fused), weight lambda/K"]
    L0 -.->|"CE gradient"| W
    LK -.->|"CE gradient"| W
    LK -.->|"KL gradient after residual cancellation"| HD1
    LK -.->|"KL gradient flows on into trunk"| TR
```

| Component | Standard MTP | KL-MTP | Soft-MTP |
|---|---|---|---|
| Output head $W$ | CE | KL | CE |
| MTP heads | CE | KL | KL (recovered) |
| Shared trunk | CE | KL | KL (recovered) |

(Table 1 in the paper.)

Each MTP head is about 3.2% of the trunk's non-embedding parameters. At 1.2B the heads total 99.1M against a 1173.9M NTP path (Appendix D).

## Level 3: execution and gradient data flow [Conceptual reconstruction from Algorithm 1 and Appendix B.4]

```mermaid
flowchart TD
    X["x: B x T"] --> E["Embeddings B x T x d"]
    E --> T["Trunk -> h0: B x T x d"]
    T --> H1["h1 = head1(h0, emb(x shifted 1)) : B x (T-1) x d"]
    H1 --> H2["h2 = head2(h1, emb(x shifted 2)) : B x (T-2) x d"]
    H2 --> H3["h3 = head3(h2, emb(x shifted 3)) : B x (T-3) x d"]
    T --> A["h0_aug[:, k:] = h0[:, k:] + s_k * (sg(hk) - hk), s_k = lambda T / (K (T-k))"]
    H1 --> A
    H2 --> A
    H3 --> A
    A --> CE0["NTP fused CE on h0_aug vs x_{t+1}"]
    H1 --> CE1["MTP fused CE vs x_{t+2}"]
    H2 --> CE2["MTP fused CE vs x_{t+3}"]
    H3 --> CE3["MTP fused CE vs x_{t+4}"]
    CE0 --> G0["dL/dh0_aug = W^T (p0 - x)  (unchanged NTP gradient to trunk)"]
    G0 --> GN["via -hk path: hk receives -g_res"]
    CE1 --> GH["hk receives g_hard"]
    CE2 --> GH
    CE3 --> GH
    GN --> GS["Sum at hk: g_hard - g_res = g_KL = W^T (p_k - q_k)"]
    GH --> GS
    GS --> BP["Backprop through heads into trunk"]
    BP --> OPT["Muon update"]
```

**Data-flow notes.**

- **Forward values are unchanged.** `sg(hk) - hk` is exactly zero, so losses and logged metrics are identical to standard MTP.
- **Backward.** The NTP loss's gradient with respect to `h0_aug` is copied, with a minus sign and scaled by $s_k$, into `hk`. This is exactly the residual term of the MTP gradient, so it cancels.
- **No logits are materialised.** Every loss uses the same fused CE kernel as standard MTP. This is why memory is 107.0 GB for both, vs 128.1 GB for explicit KL-MTP (Table 5).

## Inference path [Conceptual reconstruction]

```mermaid
sequenceDiagram
    participant P as Prompt / context
    participant M as Trunk + NTP head
    participant D as MTP heads 1..3
    P->>M: forward pass
    M->>M: sample next token y0
    M->>D: hidden states
    D->>D: draft y1, y2, y3
    D->>M: verify y1..y3 in one forward pass
    M->>M: accept prefix (speculative sampling rule), resample at first rejection
    M-->>P: append 1 + accepted tokens (avg acceptance 1.70 at 1.2B, Table 2)
```
