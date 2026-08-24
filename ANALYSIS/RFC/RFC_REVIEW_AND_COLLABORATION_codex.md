# Codex Review and Multi-Agent Collaboration Notes

## 1. Assessment of the Claude draft

The Claude draft identifies the right central thesis: synchronous and
asynchronous AFD derive value from different mechanisms. It also correctly
places operator source release inside a broader Async AFD proposal, separates
prefill delivery from decode exploration, includes an architecture diagram,
and surfaces useful package, namespace, shared expert, and license questions.

Those are strong foundations. The draft would benefit from the following
changes before community review.

### 1.1 Align the issue body with the repository RFC template

The repository template requires these top-level sections:

- Motivation
- Proposed change
- Plugin boundary
- Risks and alternatives
- Feedback period
- CC list

The Claude draft covers some of them indirectly but does not provide a complete
`Plugin boundary`, explicit risk analysis, feedback period, or reviewer list.
Using the exact template headings reduces reviewer effort and prevents required
fields from being hidden inside a long design narrative.

### 1.2 Separate the RFC issue from the detailed design

The Claude draft tries to be both documents at once and links its `Design Doc`
field back to itself. The RFC issue should contain the motivation, proposed
architecture, stable invariants, scope, decisions requested, alternatives, and
rollout. Memory protocol, chunk state, zero-token semantics, operator schemas,
and test matrices belong in a companion design document.

The Codex version therefore provides two separate drafts.

### 1.3 Use English consistently

The requested community draft is English, while the Claude draft is almost
entirely Chinese. Its technical content can be reused, but it needs a complete
English rewrite rather than a literal mixed-language translation.

### 1.4 Narrow the committed scope to prefill

Decode should be an open research direction, not Phase 2 delivery in this RFC.
The user has not selected a dynamic batching policy or interface and does not
want the current prefill protocol to freeze future decode design. The RFC title,
goals, staged delivery, and decisions requested should all be explicitly
prefill-first.

### 1.5 Keep validation language forward-looking

The RFC does not need a separate evidence-status section. It should define
reproducible correctness, reliability, compatibility, and performance
acceptance criteria without foregrounding results that are outside the
community proposal.

### 1.6 Avoid prematurely fixing source location and packaging decisions

The Claude draft proposes `csrc/npu/cam_async/`, an AFD-owned namespace, and
deletion of binary artifacts as if those choices were already accepted. The
available source tree is currently a separate `cam-aynsc` workspace root, its
Python bindings are not present yet, and license/package boundaries are still
open. These should be decision requests with a staged migration and parity
gate.

### 1.7 Keep operator-internal capacity handling below the RFC boundary

Expert-range splitting and empty-rank completion are operator or connector
correctness conditions, not feature-level RFC goals. The model runner should
observe one complete semantic work item and should not interpret internal
expert ranges. Capacity invariants and boundary tests belong in the companion
design and operator test plan.

### 1.8 Treat shared expert migration as protocol convergence work

Moving shared experts to Attention changes expert indexing, completion
conditions, operator schemas, and window sizes. It may be implemented through
a child issue, but it should not be presented as an unrelated optional
optimization because it defines the intended long-term async protocol.

### 1.9 Describe the whole feature, not only the four operators

The Async AFD solution also depends on:

- async-DP engine behavior;
- forward-context behavior without synchronized DP metadata;
- the connector-owned in-band work-item contract;
- the connector-driven FFN execution loop;
- Attention token-layout conversion and model integration;
- optional AFD-owned two-stage MoE ubatching;
- lifecycle, failure, and validation behavior.

The four operators are central data-plane primitives, not the entire feature.

### 1.10 Correct or qualify specific claims

- The source workspace is `cam-aynsc`; the public repository URL and final
  import destination should be verified before posting.
- Current source has host-side code, tiling implementations, and Ascend C
  device kernels but only a skeleton wheel binding;
  source release is not complete until the four bindings and communication
  initialization are available.
- CANN/Python/package versions should be presented as a named tested matrix,
  not automatically as permanent minimum requirements.
- The current source includes mixed license headers and derived utility code;
  license provenance requires review before choosing Apache-2.0 or MIT.
- The NVIDIA GTC 2026 reference is now available from the official
  [Groq 3 LPX technical blog](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform).

## 2. Recommended collaboration protocol

Claude and Codex do not need real-time agent-to-agent messaging. A short,
artifact-based review loop is more auditable and works even when the agents run
in separate products.

### Round 1: independent drafts

- Claude owns Markdown files whose names end in `_claude.md`.
- Codex owns Markdown files whose names end in `_codex.md`.
- Neither agent edits or renames the other agent's files.
- Both agents read the same repository state and user decisions.

Current artifacts:

```text
ANALYSIS/RFC/RFC_ASYNC_AFD_ON_ASCEND_NPU_claude.md
ANALYSIS/RFC/RFC_ASYNC_AFD_ON_ASCEND_NPU_codex.md
ANALYSIS/RFC/RFC_ASYNC_AFD_ON_ASCEND_NPU_DESIGN_codex.md
ANALYSIS/RFC/RFC_REVIEW_AND_COLLABORATION_codex.md
```

### Round 2: cross-review

Ask Claude to read the two Codex RFC/design files and write, without modifying
them:

```text
ANALYSIS/RFC/RFC_REVIEW_OF_CODEX_claude.md
```

Codex then reads that review and the latest Claude draft and writes:

```text
ANALYSIS/RFC/RFC_RESPONSE_TO_CLAUDE_codex.md
```

The response should classify every disagreement as one of:

- factual correction;
- scope decision;
- architectural trade-off;
- wording/organization;
- unresolved question requiring the user.

One cross-review round is usually enough. A second round is useful only for
unresolved architectural decisions, not stylistic preferences.

### Round 3: decision matrix

Before merging prose, the human owner should select decisions in a small matrix:

```text
Topic | Claude position | Codex position | Evidence | Owner decision
```

Important topics include prefill scope, operator source location, namespace,
shared expert ownership, operator abstraction boundary, patch/upstream boundary,
support claims, and validation criteria.

### Round 4: one final editor

Choose one agent as the final editor after decisions are made. That agent
creates the community-ready issue body and companion design document from the
decision matrix. The other agent performs a final factual review only. This
avoids two agents repeatedly rewriting the same prose.

## 3. Git coordination recommendation

Separate files in the same working tree are safer than asking two concurrent
agents to switch branches in one checkout. A `git checkout` performed by one
agent changes the working tree for the other agent and can invalidate its
assumptions.

Use one of these approaches:

1. **Recommended for the current work:** one branch, separate `_claude.md` and
   `_codex.md` files, no commits required until the user reviews them.
2. **For branch isolation:** create separate Git worktrees first, then give each
   agent ownership of one worktree and branch.
3. **For review after completion:** commit both independent artifacts, compare
   with `git diff --no-index` or a normal review PR, and merge only the chosen
   final version.

Do not have two agents create commits, switch branches, or cherry-pick in the
same working directory concurrently.

## 4. Suggested prompt for Claude

The following handoff is sufficient:

```text
Read the Codex artifacts under ANALYSIS/RFC whose names end in _codex.md. Do not
modify them. Compare them with RFC_ASYNC_AFD_ON_ASCEND_NPU_claude.md and write
your review to ANALYSIS/RFC/RFC_REVIEW_OF_CODEX_claude.md. Focus on factual
disagreements, scope, architecture, community RFC fit, and missing evidence.
For each issue, state whether you recommend accepting the Codex text, keeping
the Claude text, or asking the user to decide.
```
