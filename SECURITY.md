# Security Policy

## Reporting a vulnerability

Open a private security advisory via GitHub (Security, then Advisories) or email the
maintainer. Please do not open public issues for vulnerabilities.

## Controls in this repository

- **Secret scanning:** gitleaks (pre-commit) + GitHub secret scanning.
- **SAST:** CodeQL on every push to `main` and weekly (`.github/workflows/codeql.yml`).
- **Dependencies:** versions are pinned in the lockfile; updates are reviewed and applied by hand.
- **Secrets:** provided via `.env` (never committed); see `.env.example`.

For LLM-specific threats (prompt injection, jailbreak, data exfiltration), see
[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).
