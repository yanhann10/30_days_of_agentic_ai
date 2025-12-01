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
