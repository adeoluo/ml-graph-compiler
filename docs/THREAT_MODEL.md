# Threat Model - Ml Graph Compiler

## Assets
- What's worth protecting (data, keys, model, user trust, budget).

## Trust boundaries
- Where untrusted input crosses into the system (reference the architecture diagram).

## Threats (STRIDE + LLM-specific)

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Prompt injection | Untrusted content in context / tools | Data exfiltration, unsafe actions | Input/output filtering, tool allow-list, least privilege |
| Jailbreak | Adversarial user prompt | Policy bypass | System-prompt hardening, output classifier, refusal evaluation |
| Data exfiltration | Model echoes secrets / PII | Privacy breach | Secret scanning, output redaction, no secrets in context |
| Excessive cost | Unbounded loops / large inputs | $ blowout | Token/step budgets, rate limits, timeouts |
| Tool abuse | Agent calls dangerous tool | Destructive action | Approval gates, sandboxing, dry-run |
| Supply chain | Malicious dependency | Code execution | Dependencies pinned in `uv.lock`; CodeQL in continuous integration |

## Residual risk
- What remains after mitigations and why it's acceptable.
