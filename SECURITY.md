# Security Policy

## Scope

This project is a client-side React application with no backend, no authentication, no user data storage, and no external network requests. All computation runs in the browser.

The practical attack surface is limited to dependency vulnerabilities and supply chain issues in the npm packages used for the build tooling and UI.

## Supported Versions

| Version | Supported |
|---|---|
| `main` branch | Yes |
| Older releases | No |

Security fixes are applied to `main` only.

## Reporting a Vulnerability

If you discover a vulnerability — particularly in a dependency that could affect users who build and serve this project — please report it privately rather than opening a public issue.

**How to report:**

1. Open a [GitHub Security Advisory](https://github.com/MrZoller/neural-viz/security/advisories/new) on this repository.
2. Describe the vulnerability, affected versions, and any known impact.

You will receive a response within 7 days. If the issue is confirmed, a patch will be released as quickly as practical and credited to the reporter unless anonymity is requested.

## Dependency Policy

Runtime dependencies are kept minimal by design:

- `react` and `react-dom` — UI rendering
- `recharts` — loss curve chart

Build and dev dependencies (Vite, Tailwind, PostCSS) are not included in any served artifact. Run `npm audit` to check for known vulnerabilities in the current lockfile.
