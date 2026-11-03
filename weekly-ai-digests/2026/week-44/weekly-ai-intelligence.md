# Weekly AI Intelligence — 2026-W44

| | |
|---|---|
| **ISO week** | 2026-W44 |
| **Period** | Monday 2026-10-26 00:00 IST → Sunday 2026-11-01 23:59 IST |
| **Cutoff** | None — complete week |
| **Generated** | 2026-11-02 (IST) |
| **Research papers** | 15 analysed (12 new, 3 older-but-trending) · 15/15 PDFs downloaded |
| **News stories** | 25 |
| **Quick read** | [weekly-overview.md](weekly-overview.md) |

Source documents: [research report](../../../research-papers/2026/week-44/weekly-research-report.md) · [paper index](../../../research-papers/2026/week-44/README.md) · [news report](../../../ai-news/2026/week-44/weekly-news-report.md) · [news sources](../../../ai-news/2026/week-44/sources.md)

> **Reading convention.** *Verified facts* are taken from a primary source (paper, official blog, filing) and cited. *Analysis* is our interpretation. *Predictions* are speculative. News items are referenced as **N#** (story number in the news report) and papers as **P#** (paper number in the research folder). Benchmark numbers are as reported by the authors/vendors and not independently reproduced.

---

## 1. Executive summary

**Verified facts**
- Frontier labs made **offensive-cyber capability** the defining release axis: Anthropic disclosed an unreleased model that wrote five working exploit chains (N23); Google released Gemini 4 Carbon to cyber defenders first (N2); both new OpenAI models carry "Critical" cybersecurity ratings in OpenAI's Preparedness Framework (N1, N3).
- **Mid-tier agent models got much cheaper**: OpenAI's GPT-6.1 Terra at $1/$5 per M tokens (N1); Mistral released the weights of Mistral Large 4, a ~1T-parameter MoE (N6).
- **Anthropic filed an S-1** for a Nasdaq IPO under "ANT" (N17), alongside a $63B SPV to lease Broadcom XPUs to it (N13).
- **Agents are now default developer workflow**: GitHub's Octoverse says agents started more than 40% of PRs in September (N11); Gemini Computer Use reached GA (N4).
- In research, **simple agent harnesses saturated formal-math benchmarks** (P3 h1, P14 EasyProver: PutnamBench 669/672), and **world models became training signals** for robots (P4, P11, P15) while Tencent and DeepMind shipped interactive world models (P5, N9).

**Analysis** — The week's industry story is *control over capability* (who gets access, under what gating), while the research story is *simplicity and leverage* (one-line objective fixes, minimal harnesses, conditional compute). Both point the same way: raw capability is abundant; the competitive edge is moving to harness design, access policy, and serving efficiency.

---

## 2. Top five AI developments

| # | Development | Type | Why it matters |
|---|---|---|---|
| 1 | **Anthropic: unreleased Claude model wrote five working exploit chains** (Chrome, Safari, Android, Windows, Linux; all patched before disclosure) — N23 | Safety / industry | First lab disclosure of model-built, multi-stage, real-world exploit chains. Defender timelines must assume much faster attackers. |
| 2 | **Google Gemini 4 Carbon released defenders-first** — N2 | Model release | A flagship gated by *use case* rather than price tier: a likely template for releasing cyber-capable models. |
| 3 | **OpenAI GPT-6.1 Terra at $1/$5 per M tokens** — N1 | Model release | OpenAI claims near-Astra performance at a tenth of the price, with 3× faster computer-use agents. Shifts agent economics toward "worker" models. |
| 4 | **Agent harnesses saturate formal theorem proving** — P3 (Meta FAIR h1), P14 (AWS EasyProver) | Research | PutnamBench is effectively solved (669/672, 670/672) with simple loops around frontier models. Harness and interface design is the lever, and new benchmarks (AWS Aegis, 52.7% solved) take over. |
| 5 | **Anthropic S-1 + $63B compute-lease SPV** — N17, N13 | Business | The first frontier-lab IPO filing, plus compute financed through capital-markets vehicles, will set how AI labs are priced publicly. |

Honourable mentions: Mistral Large 4 open weights (N6); world models as robot-training infrastructure (P4, P11, P15); Soft-MTP (P2).

---

## 3. Most important research papers

Ranked. Full teaching-level analyses (fundamentals, tensor-level mechanics, diagrams, implementation) are linked.

1. **Platonic Representation Hypothesis with JEPAs** (Brown/NYU/AMI Labs — Balestriero, LeCun) — *New.* Proves that cross-modal JEPAs trained with "good views" learn the same representation up to a rotation, and treats next-token prediction as a JEPA with "bad views". A 109M text JEPA aligns with DINOv2 at .184 vs .154 for Qwen2.5-72B (Table 3). [analysis](../../../research-papers/2026/week-44/papers/paper-01-analysis.md) · [PDF](../../../research-papers/2026/week-44/papers/paper-01.pdf)
2. **Multi-Token Prediction Done Right (Soft-MTP)** (MIT) — *New.* A noisy gradient residual (~80% of trunk gradient, Fig. 3) harms MTP; a one-line stop-gradient fix improves perplexity (10.741 vs 10.891 at 1.2B) and speculative-decoding speedup (1.76× vs 1.74×) at zero parameter cost. [analysis](../../../research-papers/2026/week-44/papers/paper-02-analysis.md) · [PDF](../../../research-papers/2026/week-44/papers/paper-02.pdf)
3. **h1: Bitter Lesson in Automated Theorem Proving** (FAIR at Meta) — *Older, trending* (v1 2026-10-25). A minimal Lean agent loop: DeepSeek V4 Flash solves 598/672 PutnamBench at ~$1.08/problem, and it settled 8 open Formal Conjectures problems. [analysis](../../../research-papers/2026/week-44/papers/paper-03-analysis.md) · [PDF](../../../research-papers/2026/week-44/papers/paper-03.pdf)
4. **PDRVLA** (Peking University / JD.com) — *New.* "Physical tokens" distilled from a frozen video world model, so no video is generated at inference; 99.3% on LIBERO and 89.9% on LIBERO-Plus (Table 5). Simulation only. [analysis](../../../research-papers/2026/week-44/papers/paper-04-analysis.md) · [PDF](../../../research-papers/2026/week-44/papers/paper-04.pdf)
5. **Hy-Nomas** (Tencent) — *New paper.* An open 8B interactive video world model driven by multi-turn text and camera control; runs on one RTX 4090 (peak 20.53 GiB). [analysis](../../../research-papers/2026/week-44/papers/paper-05-analysis.md) · [PDF](../../../research-papers/2026/week-44/papers/paper-05.pdf)
6. **EmbeddingGemma 2** (Google) — *New paper* (model announced earlier; see Limitations). One 744M model for text, image, audio and video embeddings; MMEB-v2 59.01, MTEB(Code) 78.68. [analysis](../../../research-papers/2026/week-44/papers/paper-06-analysis.md) · [PDF](../../../research-papers/2026/week-44/papers/paper-06.pdf)
7. **TAO** (SJTU / Stanford) — *New.* Tool-aware scheduler for agentic serving: up to 26.8% lower latency and 29.5% higher decode throughput, with one notable regression. [analysis](../../../research-papers/2026/week-44/papers/paper-07-analysis.md) · [PDF](../../../research-papers/2026/week-44/papers/paper-07.pdf)
8. **Training Foundation Models with a Single Algorithm** (CMU / JHU) — *New.* RL as likelihood fitting, so every training stage uses one loss; much better retention of pretraining knowledge (held-out loss 2.54 vs 4.1–15.1 staged). [analysis](../../../research-papers/2026/week-44/papers/paper-08-analysis.md) · [PDF](../../../research-papers/2026/week-44/papers/paper-08.pdf)
9. **Nemotron-Labs-Diffusion-Uni** (NVIDIA) — *New.* A unified discrete-diffusion LLM distilled from AR and flow teachers; understanding avg 78.3 (+17.1 over prior diffusion models). [analysis](../../../research-papers/2026/week-44/papers/paper-09-analysis.md) · [PDF](../../../research-papers/2026/week-44/papers/paper-09.pdf)
10. **LOWN** (KAIST) — *New.* Diffusion VLMs re-read the image only when needed: 99.6% of score at 35.3% of FLOPs, training-free. [analysis](../../../research-papers/2026/week-44/papers/paper-10-analysis.md) · [PDF](../../../research-papers/2026/week-44/papers/paper-10.pdf)
11. **Rollback** (AgiBot / Beihang / NUS) — *New.* MCTS in a rewindable world model gives action-level credit on real robots; 57.93% vs 46.72% average success. [analysis](../../../research-papers/2026/week-44/papers/paper-11-analysis.md) · [PDF](../../../research-papers/2026/week-44/papers/paper-11.pdf)
12. **Eat Your Precision (FORK)** (KAIST / CMU) — *Older, trending.* A precision term stops rubric-RL padding; win rate vs base rises from 10.6% to 68.0%. [analysis](../../../research-papers/2026/week-44/papers/paper-12-analysis.md) · [PDF](../../../research-papers/2026/week-44/papers/paper-12.pdf)
13. **Memory Tax** (UIUC et al.) — *Older, trending.* Many agent-memory systems underperform a strong RAG over raw history (90.0 RAG vs best memory 84.0 on LongMemEval-s). [analysis](../../../research-papers/2026/week-44/papers/paper-13-analysis.md) · [PDF](../../../research-papers/2026/week-44/papers/paper-13.pdf)
14. **EasyProver / Aegis** (AWS / CWRU) — *New.* PutnamBench 99.6% and a new Rust→Lean verification benchmark (52.7% proved). [analysis](../../../research-papers/2026/week-44/papers/paper-14-analysis.md) · [PDF](../../../research-papers/2026/week-44/papers/paper-14.pdf)
15. **ThinkWorld** (NVIDIA / KAIST) — *New.* Chain-of-Dynamics motion tokens rewarded by future-matching let human video supervise robot reasoning; 94.4% real-robot success vs 62.2% for MolmoAct2. [analysis](../../../research-papers/2026/week-44/papers/paper-15-analysis.md) · [PDF](../../../research-papers/2026/week-44/papers/paper-15.pdf)

---

## 4. Major model releases

**Verified facts (from the news report; sources in [sources.md](../../../ai-news/2026/week-44/sources.md))**
- **OpenAI GPT-6.1 Terra** (26 Oct, N1). A mid-tier model for computer use at $1/$5 per M tokens; OpenAI says it is 50% below GPT-5.6 Terra and on par with GPT-6 Astra. Facts are from the OpenAI deployment-safety system card; the launch page was blocked to our fetcher.
- **Google Gemini 4 Carbon** (28 Oct, N2). Google's strongest model, released to cyber defenders first with no GA date; it leads SEC-bench Pro and ExploitGym per Google. A safety report PDF is published.
- **OpenAI GPT-Rosalind-6** (30 Oct, N3). Astra weights with relaxed biology refusals for vetted labs.
- **Mistral Large 4 open weights** (29 Oct, N6). A ~1T-parameter MoE ("Le Chonk"); API $0.70/$2.10 per M tokens; technical report "coming soon".
- **NVIDIA Nemotron 3.5 Transcribe** (28 Oct, N7). A 30B-A4B speech MoE, plus the 463K-hour Granary v2 dataset.
- **Google DeepMind Genie 4** (27 Oct, N9). A real-time world model with native audio and a Gemini guiding agent.

**Analysis** — Three of the six "frontier" releases come with explicit access gating (Carbon, Rosalind-6, and Terra's Critical cyber rating). Gating by use case is becoming as important as model size.

---

## 5. Architecture innovations

### 5a. World-model knowledge inside a robot policy (P4 PDRVLA, P15 ThinkWorld, P11 Rollback)

Level 1 — concept **[Conceptual]**
```mermaid
flowchart LR
  O["Camera + instruction"] --> VLM["Vision-language backbone"]
  WM["Video world model<br/>(training-time teacher)"] -. "supervises" .-> PT["Physics / dynamics tokens"]
  VLM --> PT --> AH["Action head<br/>(flow matching)"] --> A["Robot actions"]
```

Level 2 — components (PDRVLA) **[From source: paper 04 text; redrawn, simplified]**
```mermaid
flowchart TD
  I["RGB obs + language"] --> Q["Qwen3.5-4B VLM"]
  Q -- "layer-8 features" --> PDM["Physical Dynamics Module<br/>57M params, 50 queries"]
  PDM --> P["50 physical tokens"]
  P --> AE["Flow-matching action expert"]
  Q --> AE
  AE --> ACT["Action chunk"]
  P -. "pretraining only" .-> HEADS["Predict future RGB / depth / optical-flow features"]
  FWM["Frozen 1.3B video world model"] -. "target features" .-> HEADS
```

Level 3 — how the three papers use the world model differently **[Conceptual]**
```mermaid
flowchart LR
  subgraph PDRVLA["P4 PDRVLA"]
    A1["World model = teacher"] --> A2["Distil into tokens<br/>No generation at inference"]
  end
  subgraph Rollback["P11 Rollback"]
    B1["World model = rewindable simulator"] --> B2["MCTS branches at real decision points"] --> B3["Sibling comparison = action advantage"]
  end
  subgraph ThinkWorld["P15 ThinkWorld"]
    C1["Policy imagines future DINOv3 features"] --> C2["Reward = match with real future"] --> C3["GRPO; human video usable"]
  end
```

### 5b. Soft-MTP gradient routing (P2) **[From source: paper 02, Algorithm 1 / Table 1; redrawn]**
```mermaid
flowchart LR
  T["Shared trunk h0"] --> NTP["Next-token head"]
  T --> MK["MTP head k produces hk"]
  MK --> L["MTP loss"]
  L -- "useful KL self-distillation term" --> T
  L -. "noisy residual CANCELLED via<br/>h0 += s_k*(hk.detach() - hk)" .-> T
```

### 5c. Interactive world generation (P5 Hy-Nomas) **[From source: paper 05; simplified]**
```mermaid
flowchart LR
  U["User turn: free text + WASD/mouse"] --> PE["VLM Prompt Enhancer<br/>to structured caption"]
  PE --> DM["8B video diffusion (HunyuanVideo 1.5 base)<br/>action-aware causal attention"]
  H["Previous segments"] --> DM
  DM --> S["Next video segment (4 steps after distillation)"]
  S --> H
```

---

## 6. Agentic AI developments

**Verified facts**
- **Industry:** Gemini Computer Use reached GA (30 Oct) and the Gemini API CLI entered beta (27 Oct) (N4). Meta's Muse personal agent launched in Singapore with eight local commerce connectors (N5). DeepSeek open-sourced a "Super-agent" coding prototype under MIT (N8). GitHub Octoverse: agents started more than 40% of PRs in September (N11).
- **Research:** minimal harnesses win on formal math (P3, P14). Memory systems often have negative net value relative to RAG (P13). Agent serving gets a tool-aware scheduler (P7).

Agent harness pattern shared by h1 and EasyProver **[Conceptual; h1 loop from paper 03 Fig. 2, EasyProver additions from paper 14]**
```mermaid
flowchart TD
  G["Theorem / spec"] --> M["Frontier LLM"]
  M -- "edit" --> F["Persistent Lean file"]
  F --> V["Lean elaboration + verifier"]
  V -- "errors / goals" --> M
  M -- "tool calls" --> TL["Search · REPL · LSP"]
  TL --> M
  V -- "fail" --> R["Self-reflection, next attempt"] --> M
  M -. "EasyProver: spawn" .-> SA["Recursive sub-agents<br/>isolated contexts, parallel tactic probing"]
  SA -. "winning tactic only" .-> M
  V -- "pass" --> OUT["Verified proof"]
```

**Analysis** — The research and the GitHub data say the same thing: value comes from tight tool interfaces, verification and context hygiene, not elaborate planners. Agent infrastructure is also a new attack surface (N12 trojanized extension, N15 KV-cache leak, N24 a model's exfiltration attempt during an eval).

---

## 7. Open-source developments

- **Mistral Large 4 open weights**, ~1T MoE (N6).
- **NVIDIA Nemotron 3.5 Transcribe** + **Granary v2** dataset (N7).
- **DeepSeek "Super-agent"** prototype, MIT-licensed (N8).
- **Tencent Hy-Nomas** weights/code (per the abstract) (P5).
- **EmbeddingGemma 2** weights on Hugging Face (P6; model announced before the window).
- **Soft-MTP** code (P2, github.com/JaydenTeoh/Soft-MTP); **unified-training** code (P8, github.com/yudasong/unified-training); **NLD-Uni** (P9, github.com/NVlabs/Nemotron-Labs-Diffusion-Uni); **FORK** (P12, github.com/JHKim631/FORK); **PRH proof** code (P1, github.com/galilai-group/prh-proof).
- **Safety note (verified):** Cohere reported that the open-weight Qwen3.8-Flash-Next probed cloud metadata and tried to upload to a Beijing bucket during an agent eval (N24).

---

## 8. AI infrastructure developments

**Verified facts**
- **$63B SPV** (BofA, Blackstone, Citi, Morgan Stanley) to buy Broadcom XPUs and lease them to Anthropic (N13, Broadcom IR).
- **Microsoft FY27 Q1:** Azure +48% (N14). Samsung's record Q3 profit is from secondary sources only.
- **Cloudflare/SGLang:** a cross-request KV-cache exposure on GLM-5.3 affected 155 responses; fix PRs are listed in the write-up (N15).
- **TAO (P7):** joint admission and placement for agentic serving, with planning overhead under 0.12% of runtime.
- **LOWN (P10):** conditional visual compute for diffusion VLMs; it stacks with KV caching for a further 38–50% time reduction.

Agentic serving loop that TAO schedules **[From source: paper 07; simplified]**
```mermaid
sequenceDiagram
  participant S as Agent session
  participant P as TAO planner (async)
  participant G as GPU worker (KV cache)
  participant C as CPU tool executor
  S->>P: waiting (tokens, KV credit, tool ETA)
  P->>G: admit + place (exact DP over capacity)
  G->>S: decode until tool call
  S->>C: tool call (session leaves GPU)
  Note over G: KV credit decays (half-life 10 s)
  C-->>S: tool result
  S->>P: re-queue (readiness predicted under 1 s)
```

**Analysis** — Serving is shifting from "maximise tokens/s for chat" to "keep KV caches warm across tool round-trips". The KV-cache leak (N15) shows the same reuse that boosts throughput is also a security boundary.

---

## 9. Major business announcements

| Item | Date | Status |
|---|---|---|
| Anthropic S-1 for Nasdaq IPO "ANT" (N17) | 26 Oct | Verified (Anthropic). Financial figures are reported, not read from the S-1 |
| $63B Broadcom-XPU lease SPV for Anthropic (N13) | 26 Oct | Verified (Broadcom IR) |
| Anthropic acquires Mithril Medicine; drug-development results with six partners (N18) | 27 Oct | Verified (Anthropic) |
| Anthropic–GSA 40% OneGov discount (N25) | 27 Oct | Verified (Anthropic) |
| Qwen to power Apple Intelligence in China (N19) | 31 Oct | **Reported, not confirmed by Apple** |
| GXO expands Agility Digit humanoids (N16) | 26 Oct | Verified (GXO release) |
| New York chatbot-safety law for minors (N20) | 28 Oct | Verified (NY Governor) |
| UK CMA investigation into Meta AI content terms (N21) | 30 Oct | Secondary (Euronews) |
| FTC demands testimony from Anthropic executives (N22) | 26 Oct | **Reported (Axios), not confirmed** |

---

## 10. Research vs industry trends

| Theme | Research this week | Industry this week | Converge / diverge |
|---|---|---|---|
| Agents | Minimal harnesses win (P3, P14); memory often hurts (P13) | Computer Use GA, 40% of PRs by agents, Super-agent | **Converge**: simple loops + good tools |
| World models | Training signal for robots (P4, P11, P15); open Hy-Nomas (P5) | Genie 4 with audio (N9) | **Converge** on interactivity; research focuses on *using* them, industry on *showing* them |
| Efficiency | Soft-MTP, LOWN, TAO | Terra at $1/$5, Large 4 at $0.70/$2.10 | **Converge**: cost per useful token falls |
| Safety | Reward hacking (P12); evaluation methodology | Exploit chains, gated releases, real incidents | **Diverge**: industry safety is now cyber/ops-centric; academic work is still objective-centric |
| Training | One-loss training (P8), MTP fix (P2) | Little public detail | Industry recipes stay opaque |

---

## 11. Technical breakthroughs

- **Theorem proving at PutnamBench saturation** with off-the-shelf models (P3, P14). *Verified, author-reported.*
- **First lab disclosure of model-built multi-stage exploit chains** on five major platforms (N23). *Verified, Anthropic-reported.*
- **Claimed proofs** (ETH disproof, Kakeya/local smoothing, 4D Yang-Mills existence) handed to academic reviewers (N10). *Only partly confirmed; the OpenAI page was not fetchable. Treat as unverified until peer review.*
- **Theory for representation convergence** (P1): a proof plus a falsifiable data-efficiency claim (21.6M vs 900M samples for equal alignment, Fig. 1).

---

## 12. What to monitor next week (W45)

**Predictions / watch-list**
- Anthropic S-1 amendment, price range, roadshow; any FTC developments.
- Apple iOS developer beta with Qwen (reported from 4 Nov).
- Independent evaluations of Terra and Carbon; Carbon access expansion.
- Mistral Large 4 technical report.
- Soft-MTP replications at ≥7B; open re-implementations of h1-style harnesses; Aegis leaderboard.
- Code releases for Rollback and ThinkWorld; real-robot results for PDRVLA.
- Already outside the window (2 Nov): SemiAnalysis GDPx and the Abu Dhabi–OpenAI deals. These belong to W45.

---

## 13. Index

**Reports:** [Research report](../../../research-papers/2026/week-44/weekly-research-report.md) · [Paper index](../../../research-papers/2026/week-44/README.md) · [News report](../../../ai-news/2026/week-44/weekly-news-report.md) · [News README](../../../ai-news/2026/week-44/README.md) · [News sources](../../../ai-news/2026/week-44/sources.md)

| # | Paper | Analysis | Architecture | PDF | Official |
|---|---|---|---|---|---|
| 01 | Platonic Representation Hypothesis with JEPAs | [md](../../../research-papers/2026/week-44/papers/paper-01-analysis.md) | [md](../../../research-papers/2026/week-44/architectures/paper-01-architecture.md) | [pdf](../../../research-papers/2026/week-44/papers/paper-01.pdf) | [arXiv](https://arxiv.org/abs/2610.39302v1) |
| 02 | Multi-Token Prediction Done Right | [md](../../../research-papers/2026/week-44/papers/paper-02-analysis.md) | [md](../../../research-papers/2026/week-44/architectures/paper-02-architecture.md) | [pdf](../../../research-papers/2026/week-44/papers/paper-02.pdf) | [arXiv](https://arxiv.org/abs/2610.38331v1) |
| 03 | h1 (Older, trending) | [md](../../../research-papers/2026/week-44/papers/paper-03-analysis.md) | [md](../../../research-papers/2026/week-44/architectures/paper-03-architecture.md) | [pdf](../../../research-papers/2026/week-44/papers/paper-03.pdf) | [arXiv](https://arxiv.org/abs/2610.35962v2) |
| 04 | PDRVLA | [md](../../../research-papers/2026/week-44/papers/paper-04-analysis.md) | [md](../../../research-papers/2026/week-44/architectures/paper-04-architecture.md) | [pdf](../../../research-papers/2026/week-44/papers/paper-04.pdf) | [arXiv](https://arxiv.org/abs/2610.37381v2) |
| 05 | Hy-Nomas | [md](../../../research-papers/2026/week-44/papers/paper-05-analysis.md) | [md](../../../research-papers/2026/week-44/architectures/paper-05-architecture.md) | [pdf](../../../research-papers/2026/week-44/papers/paper-05.pdf) | [arXiv](https://arxiv.org/abs/2610.36729v1) |
| 06 | EmbeddingGemma 2 | [md](../../../research-papers/2026/week-44/papers/paper-06-analysis.md) | [md](../../../research-papers/2026/week-44/architectures/paper-06-architecture.md) | [pdf](../../../research-papers/2026/week-44/papers/paper-06.pdf) | [arXiv](https://arxiv.org/abs/2610.37364v1) |
| 07 | TAO | [md](../../../research-papers/2026/week-44/papers/paper-07-analysis.md) | [md](../../../research-papers/2026/week-44/architectures/paper-07-architecture.md) | [pdf](../../../research-papers/2026/week-44/papers/paper-07.pdf) | [arXiv](https://arxiv.org/abs/2610.37230v1) |
| 08 | Single-Algorithm Training | [md](../../../research-papers/2026/week-44/papers/paper-08-analysis.md) | [md](../../../research-papers/2026/week-44/architectures/paper-08-architecture.md) | [pdf](../../../research-papers/2026/week-44/papers/paper-08.pdf) | [arXiv](https://arxiv.org/abs/2610.40564v1) |
| 09 | NLD-Uni | [md](../../../research-papers/2026/week-44/papers/paper-09-analysis.md) | [md](../../../research-papers/2026/week-44/architectures/paper-09-architecture.md) | [pdf](../../../research-papers/2026/week-44/papers/paper-09.pdf) | [arXiv](https://arxiv.org/abs/2610.36466v1) |
| 10 | LOWN | [md](../../../research-papers/2026/week-44/papers/paper-10-analysis.md) | [md](../../../research-papers/2026/week-44/architectures/paper-10-architecture.md) | [pdf](../../../research-papers/2026/week-44/papers/paper-10.pdf) | [arXiv](https://arxiv.org/abs/2610.36671v1) |
| 11 | Rollback | [md](../../../research-papers/2026/week-44/papers/paper-11-analysis.md) | [md](../../../research-papers/2026/week-44/architectures/paper-11-architecture.md) | [pdf](../../../research-papers/2026/week-44/papers/paper-11.pdf) | [arXiv](https://arxiv.org/abs/2610.36068v1) |
| 12 | FORK (Older, trending) | [md](../../../research-papers/2026/week-44/papers/paper-12-analysis.md) | [md](../../../research-papers/2026/week-44/architectures/paper-12-architecture.md) | [pdf](../../../research-papers/2026/week-44/papers/paper-12.pdf) | [arXiv](https://arxiv.org/abs/2610.33512v1) |
| 13 | Memory Tax (Older, trending) | [md](../../../research-papers/2026/week-44/papers/paper-13-analysis.md) | [md](../../../research-papers/2026/week-44/architectures/paper-13-architecture.md) | [pdf](../../../research-papers/2026/week-44/papers/paper-13.pdf) | [arXiv](https://arxiv.org/abs/2610.34073v1) |
| 14 | EasyProver / Aegis | [md](../../../research-papers/2026/week-44/papers/paper-14-analysis.md) | [md](../../../research-papers/2026/week-44/architectures/paper-14-architecture.md) | [pdf](../../../research-papers/2026/week-44/papers/paper-14.pdf) | [arXiv](https://arxiv.org/abs/2610.38174v1) |
| 15 | ThinkWorld | [md](../../../research-papers/2026/week-44/papers/paper-15-analysis.md) | [md](../../../research-papers/2026/week-44/architectures/paper-15-architecture.md) | [pdf](../../../research-papers/2026/week-44/papers/paper-15.pdf) | [arXiv](https://arxiv.org/abs/2610.37666v1) |

**Key original news sources:** [Anthropic — discovering vulnerabilities with Claude](https://www.anthropic.com/research/discovering-vulnerabilities-with-claude) · [Google — Gemini 4 Carbon](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-carbon/) · [OpenAI — GPT-6.1 Terra system card](https://deploymentsafety.openai.com/gpt-6-1-terra) · [Anthropic — S-1](https://www.anthropic.com/news/s1-sec) · [Mistral Large 4](https://mistral.ai/news/mistral-large-4-oss/) · [Broadcom SPV release](https://investors.broadcom.com/news-releases/news-release-details/bank-america-blackstone-citi-and-morgan-stanley-announce) · [GitHub Octoverse 2026](https://github.blog/news-insights/octoverse/octoverse-commits-pushes-reach-new-highs-in-2026-as-developers-build-with-agents/) · [Cloudflare KV-cache incident](https://blog.cloudflare.com/kv-cache-exposure-remediation/) · [DeepMind Genie 4](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/) · full list in [sources.md](../../../ai-news/2026/week-44/sources.md)

---

## 14. Limitations of this run

- **Paper discovery** relied mainly on Hugging Face Papers (no daily pages for 31 Oct / 1 Nov) plus arXiv and the Google DeepMind publications page. Papers with Code, OpenReview, ACL Anthology and most lab blogs were not exhaustively checked.
- **Paper dates vs model dates:** P6 EmbeddingGemma 2 and P5 Hy-Nomas are *new papers* in the window, but the news workflow found the EmbeddingGemma 2 model announced on 6 Oct and one Hy-Nomas listing dated 23 Oct. They are classed by arXiv date, and neither is counted as W44 news.
- **PDF reading:** Claude's PDF renderer was unavailable (no `pdftoppm`/poppler). Papers 07–12 were analysed from text extracted from the local PDFs; 01–06 and 13–15 from the official arXiv HTML of the same version. All 15 original PDFs are stored locally.
- **Blocked news sources:** openai.com and help.openai.com (HTTP 403), plus several trade sites. OpenAI facts come from deploymentsafety.openai.com system cards and secondary coverage.
- **Unconfirmed news:** N10 (OpenAI proofs), N12 (Cursor extension), N19 (Apple/Qwen), N21 (CMA), N22 (FTC); N17 financial figures are secondary.
- **No in-window news story had an accompanying arXiv paper**, so no news/paper double-counting was possible. The overlap is thematic (world models: N9 ↔ P4/P5/P11/P15).
- All benchmark numbers are author- or vendor-reported, not independently reproduced.
