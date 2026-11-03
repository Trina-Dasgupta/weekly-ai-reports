# Paper 14 — Architecture: EasyProver and Aegis

Back to the analysis: [../papers/paper-14-analysis.md](../papers/paper-14-analysis.md)

Provenance: the paper has Fig. 1 (asynchronous agent loop), Fig. 2 (parallel tactic probing), Fig. 3 (InfiniLean service) and Fig. 4 (Aegis pipeline). Diagrams labelled **[From paper: Fig. X — redrawn]** follow those figures' described structure. Everything else is a **[Conceptual reconstruction]**.

---

## Level 1 — Concept **[Conceptual reconstruction]**

```mermaid
flowchart LR
    SRC["Problems: math theorems, verification conditions, LeetCode tasks"] --> AEG["Aegis: generate Rust -> Lean tasks + specs"]
    SRC --> EP["EasyProver: untrained frontier model + agent harness"]
    AEG --> EP
    EP --> INF["InfiniLean: elastic Lean elaboration"]
    INF --> EP
    EP --> CMP["Lean Comparator: trusted acceptance"]
    CMP --> R["Proof of P / proof of not-P / unresolved"]
```

- **Aegis** supplies harder tasks whose obligations come from real implementation semantics.
- **EasyProver** is the search policy. All of its intelligence is in the frozen model plus the harness.
- **InfiniLean** is the compute substrate for Lean queries.
- **Comparator** is the only source of truth for success.

---

## Level 2 — Components

### 2a. EasyProver agent loop **[From paper: Fig. 1 — redrawn]**

```mermaid
flowchart TD
    R["Root agent (frozen statement, workspace W0)"] --> M["Model turn"]
    M --> TOOLS["Tools: shell/files, Lean state + diagnostics, premise retrieval, tactic probe, Agent"]
    TOOLS --> M
    TOOLS -- "Agent tool" --> K1["Child 1: fresh context, copy W1, same tools (can recurse)"]
    TOOLS -- "Agent tool" --> K2["Child 2: fresh context, copy W2"]
    K1 -- "result returned asynchronously" --> M
    K2 -- "result returned asynchronously" --> M
    M --> SUB["Submit candidate"]
    SUB --> LOCK["Promote to solution file under lock"]
    LOCK --> CHK{"Comparator accept?"}
    CHK -- yes --> HALT["Global stop: first accepted proof wins"]
    CHK -- no --> REV["Revert + feedback"] --> M
```

**Explanation.**
- Children isolate both context and filesystem, so failed explorations never pollute the parent.
- Asynchrony means one slow child cannot block progress.
- The model, not a value function, decides which branch to expand. The paper calls this only "loosely analogous" to MCTS.

### 2b. Aegis pipeline **[From paper: Fig. 4 — redrawn]**

```mermaid
flowchart LR
    LC["LeetCode problem (single pure function, first-order data)"] --> CA["Coding agent: safe Rust passing reference tests"]
    CA --> CH["Charon: Rust -> LLBC"]
    CH --> AE["Aeneas: LLBC -> Lean"]
    AE --> DV{"Differential validation: Rust == Lean on all tests"}
    DV -- pass --> SA["Spec agent: theorem with sorry, asked to hold or fail"]
    DV -- fail --> DROP["Discard"]
    SA --> FL["Filters: duplicates, trivial/vacuous, elaboration errors"]
    FL --> PL["Optional Plausible screen (no positive labels)"]
    PL --> EP["EasyProver proves P or not-P"]
```

Yield: 2,045 / 2,356 problems validated (86.8%), 11,533 candidate specifications over 1,257 translated programs.

---

## Level 3 — Low-level execution

### 3a. Parallel tactic probing **[From paper: Fig. 2 — redrawn]**

```mermaid
sequenceDiagram
    participant A as Agent context
    participant P as Probe tool
    participant L as InfiniLean
    A->>P: probe(state, [simp, omega, aesop, decide, ...])
    par concurrent probes (bounded by 16-query semaphore)
        P->>L: elaborate state + simp
        P->>L: elaborate state + omega
        P->>L: elaborate state + aesop
    end
    L-->>P: per-tactic diagnostics (about 30,000 tokens total)
    P-->>A: winning tactic only (about 30 tokens)
```

### 3b. InfiniLean elaboration service **[From paper: Fig. 3 — redrawn]**

```mermaid
flowchart TD
    CL["Agent clients (JSON-RPC: Lean source)"] --> LB["Load balancer"]
    LB --> R1["Replica 1"]
    LB --> R2["Replica 2"]
    LB --> RN["Replica N (target-tracking autoscaling on Kubernetes)"]
    R1 --> MCP["Language-server MCP interface"]
    MCP --> WS["Ephemeral workspace per request"]
    WS --> REPL["Mathlib-preloaded REPL worker (warm elaboration about 0.2 s; cold import tens of s)"]
    REPL --> RESP["Goals / errors / success"]
    RESP --> CL
```

**Why stateless.** Any replica can serve any request, which allows horizontal scaling to the bursts created by up to 64 concurrent agents and 300+ sub-agents per hard proof.

### 3c. Acceptance path **[Conceptual reconstruction]**

```mermaid
flowchart LR
    C["Candidate proof body"] --> F["Splice into frozen statement prefix"]
    F --> E["lake env lean (read-only toolchain)"]
    E --> AX["#print axioms: no sorryAx or extra axioms"]
    AX --> OK["Accepted -> label proved / refuted"]
```
