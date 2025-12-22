MVP: GitHub Action daily picks email

Setup steps for user:

1. Create GitHub Secrets in the repo settings:

   - OLMO_API_KEY: your OpenRouter/Olmo key (from .env OLMO_API_KEY)
   - EMAILJS_API_KEY: SendGrid API key (free tier available)
   - EMAIL_TO: your gmail address (e.g., yanhann10@gmail.com)
   - EMAIL_FROM: verified SendGrid sender email (you must verify this in SendGrid)

2. The workflow will run daily, call Olmo to pick top3 (prefilters out previous picks), update reading_progress.md and picks_history.json, and send a SendGrid email to EMAIL_TO.

Notes:

- You don't need to store Gmail password. The action uses SendGrid to send email, so you only provide your Gmail address as the recipient.
- If you prefer other email providers (SES, Mailgun) I can adapt the action.
- You can trigger the workflow manually via Actions -> Run workflow -> Run.

Security:

- Keep API keys in GitHub Secrets; do not commit them.
- The action commits reading_progress.md and picks_history.json back to the repo so you have a visible history.

## Email Feedback Processing Setup (for GitHub Actions)

### How Authentication Works

**CI/GitHub Actions does NOT ask for browser login.** Instead, it uses a pre-generated `refresh_token` that was obtained during a one-time local OAuth flow.

**The Flow:**

1. **Local (one-time)**: Run OAuth flow → Browser opens → You authorize → Get `refresh_token`
2. **CI (ongoing)**: Use stored `refresh_token` → Automatically get new `access_token` when needed → No browser required

### Understanding the Two Files:

1. **`gcp_cred.json`** (OAuth Client Credentials):

   - Contains: `client_id`, `client_secret`, `project_id`, `auth_uri`, `token_uri`, etc.
   - Download from [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
   - Create OAuth 2.0 Client ID → Application type: "Desktop app"
   - **Store as GitHub Secret: `GCP_OAUTH_CREDENTIALS_JSON`** (full JSON content)

2. **`gmail_token.pickle`** (User Authorization Token):
   - Contains: `access_token` (expires in ~1 hour), `refresh_token` (long-lived), plus client info
   - Generated AFTER you authorize the app locally (one-time browser OAuth flow)
   - The `refresh_token` allows automatic token renewal without re-authorization
   - **Store as GitHub Secret: `GMAIL_TOKEN_PICKLE_B64`** (base64-encoded) or `GMAIL_TOKEN_JSON` (JSON format)

### Setup Steps:

1. **Get the refresh_token (one-time, local only)**:

   ```bash
   # Place your gcp_cred.json in the project root (or set GCP_OAUTH_CREDENTIALS_JSON env var)
   python scripts/process_email_feedback.py
   # This will:
   # - Open a browser for OAuth authorization
   # - Ask you to sign in to Gmail and grant permissions
   # - Create gmail_token.pickle with access_token + refresh_token
   ```

2. **Extract the token for GitHub Secrets**:

   **Option A: Base64-encoded pickle (recommended)**

   ```bash
   # Generate base64 string from pickle file:
   base64 -i gmail_token.pickle | pbcopy  # macOS
   # or
   base64 gmail_token.pickle | pbcopy     # Linux
   ```

   Store as GitHub Secret: `GMAIL_TOKEN_PICKLE_B64`

   **Option B: JSON format**

   ```bash
   # If you have gmail_token.json instead:
   cat gmail_token.json | pbcopy  # macOS
   ```

   Store as GitHub Secret: `GMAIL_TOKEN_JSON`

3. **Create GitHub Secrets**:
   - `GCP_OAUTH_CREDENTIALS_JSON`: Full contents of `gcp_cred.json` (plain JSON)
   - `GMAIL_TOKEN_PICKLE_B64`: Base64-encoded `gmail_token.pickle` (from step 2)
   - `OPENROUTER_API_KEY`: Your OpenRouter API key (for parsing feedback)

### How It Works in CI:

1. **Token Restoration**: Code reads `GMAIL_TOKEN_PICKLE_B64` → Decodes → Loads `refresh_token`
2. **Auto-Refresh**: If `access_token` expired → Uses `refresh_token` → Gets new `access_token` automatically
3. **No Browser**: All token refresh happens via API calls, no user interaction needed

**Note**: The `refresh_token` typically doesn't expire unless you revoke access. If it does expire, you'll need to regenerate it locally and update the GitHub Secret.
