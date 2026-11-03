# Weekly AI Research Report — 2026-W44

**Period:** Monday 2026-10-26 00:00 IST → Sunday 2026-11-01 23:59 IST (complete week)
**Generated:** 2026-11-02 (IST) · **Papers:** 15 · **PDFs:** 15/15 downloaded · Index: [README.md](README.md)

All numbers below are values reported by the papers themselves (table/figure cited in each analysis). Upvote counts are Hugging Face Papers counts as displayed when fetched on 2026-11-02.

## Executive summary

- **Harnesses beat bespoke machinery.** Two theorem-proving papers (Meta FAIR's *h1*, AWS's *EasyProver*) show that a simple, efficient agent loop around a frontier model matches or beats specialised search systems — PutnamBench is now essentially saturated (EasyProver 669/672; h1 with GPT-5.5 670/672). Attention moves to harder benchmarks such as AWS's new *Aegis* (52.7% proved).
- **World models are becoming training infrastructure, not demos.** PDRVLA distils a frozen video world model into a VLA's "physical tokens"; Rollback uses a world model as a rewindable simulator for credit assignment on real robots; ThinkWorld rewards robot "thoughts" by whether imagined futures match reality; Tencent's Hy-Nomas ships an open 8B interactive world model.
- **Objective-level fixes with outsized payoff.** Soft-MTP (one line, zero parameters) removes a noisy gradient term from multi-token prediction; FORK adds precision to rubric rewards and turns RL models from *less* preferred than the base model into preferred ones; CMU frames all training stages, including RL, as one likelihood objective.
- **Healthy skepticism about agent memory.** *Memory Tax* finds many dedicated memory systems score below a well-tuned RAG over raw history.
- **Efficiency at serving time.** TAO schedules agentic sessions across GPU inference and CPU tool calls; LOWN skips re-reading the image in diffusion VLMs when it would not change the answer (99.6% of score at 35.3% FLOPs).
- **Theory with a testable claim.** Balestriero & LeCun prove a version of the Platonic Representation Hypothesis for JEPAs and show a 109M text JEPA aligning with DINOv2 better than 72B LLMs.

## Ranked papers

### 1. Proving and Improving the Platonic Representation Hypothesis with JEPAs — *New*
Brown / NYU / AMI Labs (Balestriero, LeCun). Proves that JEPAs trained on different modalities with "good views" (two observations of the same scene) converge to the same representation up to a rotation; next-token prediction is a JEPA with "bad views", which is why LLMs only align with vision through scale. A 109M good-view text JEPA reaches .184 mutual-10NN alignment with DINOv2 vs .154 for Qwen2.5-72B (Table 3). **Why #1:** a theoretical explanation for a widely debated empirical observation, with a falsifiable data-efficiency claim. → [analysis](papers/paper-01-analysis.md) · [PDF](papers/paper-01.pdf)

### 2. Multi-Token Prediction Done Right (Soft-MTP) — *New*
MIT (sole author). Decomposes the MTP gradient into a useful self-distillation pull and a noisy residual carrying ~80% of trunk gradient (Fig. 3); Soft-MTP cancels the residual with a stop-gradient line. At 1.2B: 57.31 avg zero-shot vs 56.80 standard MTP, perplexity 10.741 vs 10.891 (Table 3). **Why here:** cheap, drop-in, broadly applicable; tested only to 1.2B. → [analysis](papers/paper-02-analysis.md) · [PDF](papers/paper-02.pdf)

### 3. h1: A Taste of the Bitter Lesson in Automated Theorem Proving — *Older, trending*
FAIR at Meta (+ EPFL, ENPC, NYU). Minimal loop: model + persistent Lean file + search/REPL/LSP tools + verifier + reflection. DeepSeek V4 Flash solves 598/672 PutnamBench at ~$1.08/problem; 8 open Formal Conjectures problems settled (Table 1). **Why here:** strong evidence that interface efficiency, not elaborate search, drives agent performance. → [analysis](papers/paper-03-analysis.md) · [PDF](papers/paper-03.pdf)

### 4. PDRVLA — *New*
Peking University / JD.com. A 57M-parameter module produces 50 "physical tokens" from Qwen3.5-4B mid-layer features, trained to predict future RGB/depth/flow features with a frozen 1.3B video world model (pretraining only). 99.3% LIBERO, 89.9% LIBERO-Plus, 93.6% RoboTwin 2.0, 81.5% SimplerEnv (Table 5) — small margins over prior best, simulation only. Most-upvoted paper of the week on HF. → [analysis](papers/paper-04-analysis.md) · [PDF](papers/paper-04.pdf)

### 5. Hy-Nomas — *New*
Tencent Hy Visual Generation Team. Open 8B video world model on HunyuanVideo 1.5, multi-turn free-form text + continuous camera control; action-aware attention masking, VLM prompt enhancer, distillation to 4 steps/segment; 100 s 1280×704 video peaks at 20.53 GiB on one RTX 4090. Human preference: +66.7 net vs LingBot-World v2 on prompt following (Fig. 13), but −16.7 vs Genie 3 on spatial memory (Fig. 17). → [analysis](papers/paper-05-analysis.md) · [PDF](papers/paper-05.pdf)

### 6. EmbeddingGemma 2 — *New*
Google. One 744M model embeds text, image, audio, video into a 768-d space; Gemma 4 decoder turned bidirectional, distilled from Gemini Embedding 2. MMEB-v2 59.01, MTEB(Code) 78.68 (Table 6); native audio retrieval 69.54 MRR@10 vs 57.59 via ASR (Table 11). Note English MTEB slightly down vs v1 (68.46 vs 69.67, Table 14). → [analysis](papers/paper-06-analysis.md) · [PDF](papers/paper-06.pdf)

### 7. TAO — *New*
Shanghai Jiao Tong / Stanford. Scheduler for agentic serving that ranks session–GPU pairs by decaying KV-cache credit, predicted tool readiness and token footprint, then solves admission + placement with an exact DP. Up to 26.8% lower latency, 36.3% lower TTFT, 29.5% higher decode throughput (Table 1) — but with a −504% TTFT regression in one AutomationBench setting (Table 3). → [analysis](papers/paper-07-analysis.md) · [PDF](papers/paper-07.pdf)

### 8. Training Foundation Models with a Single Algorithm — *New*
CMU / JHU / Aurora. Writes RL (MaxRL) as likelihood fitting to the model's own correct samples weighted 1/p̂, so pretraining, midtraining, SFT and RL become one loss and training becomes a data-mixture schedule. Better retention: held-out pretraining loss 2.537–2.540 vs 4.135–15.104 for staged (Table 11). Small scale (124M, 0.76B). → [analysis](papers/paper-08-analysis.md) · [PDF](papers/paper-08.pdf)

### 9. Nemotron-Labs-Diffusion-Uni — *New*
NVIDIA. Unified discrete-diffusion LLM (8B understanding + 8B generation towers) distilled across paradigms from AR VLMs and Qwen-Image; "self-refining" corruption allows tokens to be revised after commit. Understanding avg 78.3 (+17.1 over best prior diffusion model, Table 1); HPSv3 11.12 vs FLUX.1-dev 10.88 (Table 3). → [analysis](papers/paper-09-analysis.md) · [PDF](papers/paper-09.pdf)

### 10. LOWN — *New*
KAIST / AITRICS. Image changes the unmasked token on only 22.8% of LLaDA-V steps (Table 6); LOWN re-reads the image only when a text-only prediction disagrees with the last image-conditioned anchor. 99.6% of score at 0.353× FLOPs, training-free (Table 1). #1 on HF Papers 2026-10-30. → [analysis](papers/paper-10-analysis.md) · [PDF](papers/paper-10.pdf)

### 11. Rollback — *New*
AgiBot / Beihang / NUS. MCTS over a rewindable video world model at real-robot decision points yields action-level advantages without a value model; trains a π0.5 policy. 57.93% avg vs 46.72% for a RISE reproduction on 3 real tasks (Table 1); surpassed average human players on two tasks within 8 hours (Fig. 6). → [analysis](papers/paper-11-analysis.md) · [PDF](papers/paper-11.pdf)

### 12. Eat Your Precision (FORK) — *Older, trending*
KAIST AI / CMU LTI. Rubric-coverage rewards are recall-only; FORK adds a per-unit necessity + correctness precision term, F_β reward with GRPO on Qwen2.5-7B. Win rate vs base 10.6% → 68.0% on RubricHub-Medical; length 9,195 → 3,533 chars (Table 1). → [analysis](papers/paper-12-analysis.md) · [PDF](papers/paper-12.pdf)

### 13. Memory Tax — *Older, trending*
UIUC / Tsinghua / Renmin / Waterloo / Verdent.ai. 12 memory systems vs a strong naive RAG over raw history: on LongMemEval-s with GPT-5.6 Luna, two-round RAG scores 90.0 and no memory system beats it (best 84.0, Table 1); OpenViking loses 38.0% of facts during construction (Table 2). → [analysis](papers/paper-13-analysis.md) · [PDF](papers/paper-13.pdf)

### 14. EasyProver / Aegis — *New*
AWS / Case Western Reserve. Untrained frontier model + recursive sub-agents, async search, parallel tactic probing: PutnamBench 99.6% (669/672, Table 1), VERINA 100%. New Aegis benchmark (LeetCode → Rust → Lean): 52.7% proved. No ablations. → [analysis](papers/paper-14-analysis.md) · [PDF](papers/paper-14.pdf)

### 15. ThinkWorld — *New*
NVIDIA / KAIST. Chain-of-Dynamics motion tokens shared by human hands and robot grippers; GRPO rewards imagined futures (DINOv3 features ~1 s ahead) that match reality, so unlabeled human video supervises reasoning. Real-robot 94.4% vs 62.2% MolmoAct2 (Table 6); median latency 324.84 ms vs 879.37 ms (Table 9). → [analysis](papers/paper-15-analysis.md) · [PDF](papers/paper-15.pdf)

## New vs older-but-trending

- **New in the period (12):** 01, 02, 04, 05, 06, 07, 08, 09, 10, 11, 14, 15.
- **Older, trending (3):** 03 *h1* (v1 2026-10-25 16:37 IST; on HF Papers 2026-10-28; v2 posted 2026-10-30), 12 *FORK* (2026-10-24; HF Papers 2026-10-27), 13 *Memory Tax* (2026-10-25; HF Papers 2026-10-28).

## Themes across papers

1. **Simple loops + good interfaces > handcrafted structure** (03, 14; echoed by 13 for memory).
2. **World models as a training signal** — distillation (04), rewindable simulation for credit assignment (11), outcome-matching rewards (15), and as a product (05).
3. **Fixing objectives, not architectures** (02, 08, 12).
4. **Conditional computation at inference** (07 for agent serving, 10 for diffusion VLMs).
5. **Diffusion LLMs maturing for multimodal** (09, 10).

## Area coverage

| Area | Papers | Note |
|---|---|---|
| LLMs | 02, 08 | |
| Agentic AI | 03, 07, 13, 14 | |
| Reasoning | 03, 14, 15 | |
| Multimodal | 01, 06, 09, 10 | |
| RAG / embeddings | 06, 13 | |
| Transformer architectures | 01, 02 | |
| Training & fine-tuning | 02, 08 | |
| Reinforcement learning | 08, 11, 12, 15 | |
| Generative AI / diffusion | 05, 09, 10 | |
| AI infrastructure / distributed inference | 07 | |
| Distributed training | — | Nothing notable found |
| AI safety | 12 (indirect) | Runner-up: independent DeepSeek V4.1 Flash-1017 risk evaluation (arXiv 2610.35671) |
| Robotics | 04, 11, 15 | |
| Small language models | 06 (indirect) | Runner-up: OPTIC per-layer-embedding compression (arXiv 2610.31309) |
| Memory architectures | 13 | |
| World models | 01, 04, 05, 11, 15 | |
| AI coding systems | 14, 03 | |

**Runners-up (verified, not analysed):** Mistral *Timestral* time-series models (2610.37585), *On-Policy Self-Training* (Google, 2610.32770), *Work Agents* survey (2610.35845), *Trial by Trace* agent-harness benchmark (2610.32285).

## What to watch

- Whether Soft-MTP holds at ≥7B scale and gets adopted in open pretraining recipes.
- Independent reproductions of Aegis and of h1's cost claims; next "unsaturated" formal benchmarks.
- Real-robot and latency results for PDRVLA; code releases for Rollback and ThinkWorld.
- Community evaluations of EmbeddingGemma 2 and Hy-Nomas open weights.

## Limitations of this run

- **Discovery leaned on Hugging Face Papers** (daily pages Oct 26–30; no daily pages exist for Oct 31/Nov 1). Papers with Code, OpenReview, ACL Anthology and most lab blogs were not exhaustively checked. HF's week page is labelled "Oct 25–31", not the exact IST window.
- **PDF reading:** the Read tool could not render PDFs (no `pdftoppm`). Papers 07–12 were read from the local PDFs via text extraction; papers 01–06 and 13–15 were analysed from the official arXiv HTML of the same version (institutions for 01–03 confirmed from the local PDF title page).
- Some equations reconstructed from PDF text layers (e.g. TAO Eq. 8) or garbled HTML cells are flagged inside the individual analyses.
- No major-lab paper from OpenAI, Anthropic, Apple, DeepSeek or AI2 was found in the window.
