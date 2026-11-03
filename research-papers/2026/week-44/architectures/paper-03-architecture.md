# Architecture: h1, A Taste of the Bitter Lesson in Automated Theorem Proving (Paper 03)

Back to the analysis: [../papers/paper-03-analysis.md](../papers/paper-03-analysis.md). Raw diagram: [../diagrams/paper-03-system.mmd](../diagrams/paper-03-system.mmd).

Fig. 2 of the paper ("h1: a simple harness that scales") is the paper's own system diagram. It shows an LLM agent editing a Lean file until it submits or hits the $T$-turn limit, then validation and reflection. Level 1 below redraws that figure. Levels 2 and 3 are reconstructions from Section 2.2 and Appendix B (Table 3 tool schemas, B.2 prompts).

## Level 1: concept [From paper: Fig. 2, redrawn]

```mermaid
flowchart LR
    ST["Original Lean statement"] --> AG["LLM agent (ReAct, ≤ T turns per iteration)"]
    AG -->|"edits"| WF["Persistent Lean working file"]
    WF -->|"errors + open goals after every edit"| AG
    AG -->|"submit / forced at T"| VAL["External validator (SafeVerify)"]
    VAL -->|"pass"| OUT["Verified proof (or checked disproof)"]
    VAL -->|"fail"| REF["Reflection LLM compacts transcript + feedback"]
    REF -->|"verified progress, lessons, lemma names"| AG
```

**Components.**

- **Agent.** Any frontier LLM; no fine-tuning.
- **Working file.** Persists across iterations, so verified progress is never lost when context is reset.
- **Validator.** Rejects placeholders and added assumptions, and checks the proof against the *original* statement.
- **Reflection.** Bounds the context length and filters out defeatist or unsound reasoning.

## Level 2: components and tool interface [Conceptual reconstruction from Appendix B.1, Table 3]

```mermaid
flowchart TD
    LLM["Prover LLM + detailed prompt (tools, statement-integrity rules, do not give up)"]
    subgraph FILEOPS["Working-file tools"]
      V1["view_working_file / view_original_file (40,000-byte view)"]
      E1["edit_working_file (old, new, replace_all) -> compile feedback ≤ 32,000 chars"]
      U1["undo_edit_working_file"]
      S1["submit_working_file (updated_file, reasoning)"]
    end
    subgraph SEARCH["Library search"]
      LE["lean_leanexplore (6 results)"]
      LO["lean_loogle (6 results)"]
      GR["grep_mathlib (≤ 50 lines)"]
    end
    subgraph EXEC["Execution"]
      RP["lean_repl_run_scratch_code"]
    end
    subgraph LSP["Language server (replies ≤ 4,000 chars)"]
      G1["lean_goal / lean_term_goal"]
      D1["lean_diagnostic_messages"]
      H1["lean_hover_info / lean_declaration_file"]
      R1["lean_references (≤ 40) / lean_file_outline"]
    end
    LLM --> FILEOPS
    LLM --> SEARCH
    LLM --> EXEC
    LLM --> LSP
    S1 --> SV["Checks, then SafeVerify; ends iteration"]
    SV --> CTRL["Controller: budget B, iteration count, cost accounting"]
    CTRL --> REFL["Reflection LLM (evidence discipline, strategy discipline, preserve progress)"]
    REFL --> LLM
```

**Design notes.**

- **Tight output limits** (generic 6,000 chars, LSP 4,000, compile feedback 32,000) keep context small and cacheable.
- **Lean timeouts.** 600 s for snippets and 1,200 s for the working file. Three consecutive deterministic failures mark the problem unsolved.
- **The tool set is what matters.** Removing search, REPL and LSP drops Putnam100 from 69 to 49 and FATE-X from 63 to 49.

## Level 3: low-level execution and context/cost flow [Conceptual reconstruction]

```mermaid
flowchart TD
    START["Problem j, budget B, spent = 0"] --> CTX0["Context = system prompt + statement + summary (empty first time)"]
    CTX0 --> TURN["Turn t: LLM call with cached prefix"]
    TURN --> BILL["spent += cached-in (≈1/10) + uncached-in + out (≈10x in) token costs"]
    BILL --> TOOL{"Tool call?"}
    TOOL -->|"search / goal / REPL / edit"| OBS["Run tool, truncate output, append to context"]
    OBS --> TT{"t < T and spent < B?"}
    TT -->|"yes"| TURN
    TT -->|"no, forced submit"| SUBMIT["submit_working_file"]
    TOOL -->|"submit"| SUBMIT
    SUBMIT --> SV{"SafeVerify passes?"}
    SV -->|"yes"| DONE["Solved: record cost_j"]
    SV -->|"no"| RF["Reflection LLM: transcript + feedback -> compact block"]
    RF --> BUD{"spent < B?"}
    BUD -->|"yes"| CTX0
    BUD -->|"no"| FAIL["Unsolved within B (cost capped at B)"]
```

**Why this flow is efficient (as reported).**

- **Cached prefixes.** Within an iteration the context is append-only, so most input is cached. h1 reuses 74-94% of input from cache, vs ≤0.5% for AxProverBase.
- **Moderate T is optimal.** T = 1 gets almost no caching (0.2% cached) and loses reasoning continuity. T = 200 lets context bloat. 10 turns gave the best Putnam100/FATE-X results (72/65 vs 50/39 and 60/52; Fig. 4 right).
- **Reflection resets context** while the working file preserves state. Removing reflection drops PutnamBench (DeepSeek V4 Flash) from 598 to 554 (Fig. 6).

## Scaling axes [Conceptual reconstruction]

```mermaid
flowchart LR
    BUDGET["Test-time compute"] --> DEPTH["Depth: one long pass@1 attempt (more iterations)"]
    BUDGET --> BREADTH["Breadth: residual pass@k (fresh restarts on unsolved)"]
    BUDGET --> MODEL["Better base model in the same loop"]
    DEPTH --> R1["Open Erdős problems: pass@1 82.5% at $121 per success vs APN pass@3 75.4% at $191"]
    BREADTH --> R2["Closes remaining gaps on Putnam 2025 / PutnamBench"]
    MODEL --> R3["Newer models close gaps left by specialised workflows"]
```
