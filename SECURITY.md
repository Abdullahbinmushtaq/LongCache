# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Yes     |

Older versions are not actively maintained. Please upgrade to the latest release.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report security issues privately via one of the following:

- **GitHub Private Vulnerability Reporting** (preferred): [Security Advisories](https://github.com/ENDEVSOLS/LongCache/security/advisories/new)
- **Email**: technology@endevsols.com

### What to Include

- A clear description of the vulnerability
- Steps to reproduce or a minimal proof-of-concept
- Affected version(s) and storage backends (SQLite, Redis, Memory)
- Potential impact assessment (e.g. cache poisoning, sensitive data leakage)

### Response Timeline

| Stage | Timeframe |
|-------|-----------|
| Acknowledgement | Within 48 hours |
| Initial assessment | Within 5 business days |
| Fix or mitigation | Within 30 days (critical issues prioritized) |
| Public disclosure | After fix is released |

We follow [coordinated vulnerability disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure). You will be credited in the release notes unless you prefer to remain anonymous.

## Scope

### In Scope

- Cache poisoning vulnerabilities (cross-namespace leaks, key collision attacks)
- Remote code execution or arbitrary command execution in deserialization / backend drivers
- Denial of service in FAISS index search or embedding pipelines
- Secrets / sensitive token leaks in CLI logs or metrics exports

### Out of Scope

- Attacks requiring root / local filesystem write access to the host SQLite database
- Denial of service resulting from memory exhaustion due to user-configured unbounded limits (`max_entries`)
- Vulnerabilities in third-party LLM providers (e.g. OpenAI service outages)
