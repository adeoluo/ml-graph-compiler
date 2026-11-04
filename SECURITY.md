# Security Policy

## Reporting a vulnerability

Open a private security advisory via GitHub (Security → Advisories) or email the
maintainer. Please do not open public issues for vulnerabilities.

## Controls in this repository

- **Secret scanning:** gitleaks (pre-commit) + GitHub secret scanning.
- **SAST:** CodeQL (`.github/workflows/codeql.yml`).
- **Dependencies:** pinned in `uv.lock`. CodeQL in continuous integration.
- **Secrets:** provided via `.env` (never committed); see `.env.example`.

For LLM-specific threats (prompt injection, jailbreak, data exfiltration), see
[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).
