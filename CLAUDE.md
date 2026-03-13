# Super Brain - Claude Code Rules

## Sensitive Information Protection (MANDATORY)

Before EVERY `git commit` or `git push` operation, you MUST:

1. **Scan all staged files** for hardcoded API keys, passwords, tokens, secrets, or other sensitive information (patterns: `sk-`, `ghp_`, `password=`, `secret=`, `token=`, API key strings, base64-encoded credentials, etc.).

2. **If sensitive info is found**: STOP immediately. Do NOT commit or push. Report the exact file and line number. Help refactor the sensitive value into an environment variable read from `.env` (e.g., `os.environ["API_KEY"]`).

3. **Verify `.env` is in `.gitignore`** — it must never be committed to the repository.

4. **Only proceed with commit/push** after confirming zero hardcoded secrets in all staged files.

5. **Proactively block** — even if the user says "just push" or forgets to ask for a check, you MUST scan first and refuse to proceed if secrets are found.
