# Paper 12 architecture — FORK precision-aware rubric reward

Back to analysis: [../papers/paper-12-analysis.md](../papers/paper-12-analysis.md) · Raw diagram: [../diagrams/paper-12-system.mmd](../diagrams/paper-12-system.mmd)

FORK changes no model architecture. It is a **reward-computation pipeline** plugged into a standard GRPO loop: the reward becomes `F_β(precision, recall)` instead of rubric recall alone.

## Level 1 — Concept **[Conceptual reconstruction]** (based on Fig. 1)

```mermaid
flowchart LR
    subgraph STD["Standard rubric RL"]
        S1["Response"] --> S2["Rubric coverage (recall)"]
        S2 --> S3["Reward: more content = more boxes ticked"]
    end
    subgraph FK["FORK"]
        F1["Response"] --> F2["Recall: rubric coverage"]
        F1 --> F3["Precision: needed and correct units"]
        F2 --> F4["F_beta: both must be high"]
        F3 --> F4
    end
```

Fig. 1 of the paper shows the motivation: with the standard reward, recall saturates while precision and win rate vs the base model fall and length rises; with FORK, recall reaches a similar level while precision and win rate rise and length falls.

## Level 2 — Components **[From paper: Fig. 2]** (redrawn)

```mermaid
flowchart TD
    X["Prompt x"] --> POL["Policy pi_theta (Qwen2.5-7B-Instruct)"]
    POL --> Y["Response y"]
    subgraph GRADER["Training grader: Qwen3.6-35B-A3B"]
        UZ["Unitizer: sentence-level units, drop headers and fillers"]
        MT["Rubric-to-unit matcher m_ji"]
        TH["Threshold chooser tau"]
        FC["Factual-correctness judge C_i"]
        RS["Rubric-satisfaction judge s_j"]
    end
    Y --> UZ
    Y --> RS
    RB["Rubrics r_j, weights w_j"] --> MT
    RB --> RS
    UZ --> MT
    UZ --> FC
    MT --> CR["Credit split: a_ji = m_ji / K_j, c_i = sum_j a_ji"]
    CR --> TH
    TH --> NL["N_i = 1 if c_i >= tau"]
    NL --> PR["Precision = (1/n) sum N_i C_i"]
    FC --> PR
    RS --> RC["Recall = sum w_j s_j / sum w_j"]
    PR --> FB["r_FORK = (1+beta^2) P R / (beta^2 P + R)"]
    RC --> FB
    FB --> OPT["GRPO update (16 rollouts per prompt)"]
    OPT --> POL
```

| Component | Input | Output | Notes |
|---|---|---|---|
| Unitizer | response | units `u_1..u_n` | ≈ sentences; preserve wording; pronoun resolution only (Sec. 3.2) |
| Matcher | rubrics, units | binary matrix `m` (`M×n`) | "does unit i contribute to rubric j" |
| Credit split | `m` | credits `c_i` | each rubric distributes 1 credit equally → redundancy is diluted |
| Threshold chooser | units + credits | `τ > 0` | dynamic per response, chosen by the grader |
| Correctness judge | prompt, unit | `C_i ∈ {0,1}` | strict: any factual error → 0 |
| Satisfaction judge | response, rubric | `s_j ∈ {0,1}` | standard rubric coverage |
| F_β combiner | P, R | scalar reward | `β = 1` or `1.25` |
| GRPO | rewards for 16 samples | policy update | batch 128, lr 4.2e-6, 3 epochs, 8× H100 |

## Level 3 — Credit computation on the Fig. 2 example **[From paper: Fig. 2]** with illustrative τ and correctness

```mermaid
flowchart LR
    R1["r1 chlorophyll breaks down (w 0.45)"] -->|"1/2"| U1["u1 shorter days trigger breakdown"]
    R1 -->|"1/2"| U2["u2 loss reveals yellow and orange"]
    R2["r2 pigments revealed (w 0.35)"] -->|"1"| U2
    R3["r3 shorter days trigger (w 0.20)"] -->|"1"| U1
    U1 --> C1["c1 = 1.5"]
    U2 --> C2["c2 = 1.5"]
    U3["u3 trees shed leaves to save water"] --> C3["c3 = 0"]
    U4["u4 cold air dyes leaves red (false)"] --> C4["c4 = 0"]
    C1 --> P["With tau = 1: N = 1,1,0,0 and C = 1,1,1,0, so Precision = 2/4"]
    C2 --> P
    C3 --> P
    C4 --> P
```

Credits are from Fig. 2. The threshold value and correctness labels in the final node are illustrative assumptions for teaching.

## Where FORK sits in the RL loop **[Conceptual reconstruction]**

```mermaid
sequenceDiagram
    participant T as GRPO trainer
    participant P as Policy
    participant G as Grader service
    T->>P: prompts (batch 128)
    P-->>T: 16 responses per prompt
    T->>G: response, prompt, rubrics
    G-->>T: recall R and precision P per response
    T->>T: r = F_beta(P, R), group-normalised advantage
    T->>P: gradient step
```

Evaluation is separate: GPT-5.4 acts as a reference grader for recall, precision, F1, and rubric-free win rate vs the base model.
