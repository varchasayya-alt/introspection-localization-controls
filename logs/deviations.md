# Deviations log

Every deviation from the written plan is recorded here **before** the deviating run.
Format: date · gate · what the plan said · what we will do instead · why.

| # | Date | Gate | Plan said | Doing instead | Why |
|---|---|---|---|---|---|
| D0 | 2026-10-14 | 0 | Project root `%USERPROFILE%\IntrospectionStages`; get GitHub ID with `gh api user` | Project root is `%USERPROFILE%\OneDrive\Desktop\IntrospectionStages` (the folder this session was opened in); ID fetched from the public API `api.github.com/users/varchasayya-alt` (same `id` field) | The `%USERPROFILE%\IntrospectionStages` path does not exist; `gh` CLI is not installed. Neither affects any result. |
| D1 | 2026-10-14 | 0 (proposed for all gates; needs researcher approval) | "Use bf16 for 7B models" | Store weights in bf16, but cast each Linear layer's weight to fp32 just before its matmul; residual stream, norms and embeddings in fp32. Same mode for **every** model and rung. | Measured on this CPU (AVX2 only, no bf16 hardware): pure bf16 forward = 25.7 s vs fp32 4.4 s vs this hybrid 4.9 s (OLMo-2 1B, 300 tokens). Same RAM as bf16; numerically at least as accurate as bf16. |
