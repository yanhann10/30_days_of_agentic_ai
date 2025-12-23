import os
import requests
from dotenv import load_dotenv

load_dotenv()

EMAIL_TO = os.environ.get('EMAIL_TO')
EMAIL_FROM = os.environ.get('EMAIL_FROM')
SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY')
EMAILJS_SERVICE_ID = os.environ.get('EMAILJS_SERVICE_ID')
EMAILJS_TEMPLATE_ID = os.environ.get('EMAILJS_TEMPLATE_ID')
EMAILJS_USER_ID = os.environ.get('EMAILJS_USER_ID')
EMAILJS_PRIVATE_KEY = os.environ.get('EMAILJS_PRIVATE_KEY')

SEND_EMAIL_ENABLED = os.environ.get('SEND_EMAIL', 'true').lower() in ('true', '1', 'yes')


def send_email(subject, body_html, body_text=None):
    if not SEND_EMAIL_ENABLED:
        print("📧 Email sending is DISABLED (set SEND_EMAIL=true to enable)")
        print(f"   Subject: {subject}")
        print(f"   To: {EMAIL_TO}")
        print(f"   Preview: {body_text[:200] if body_text else body_html[:200]}...")
        return False

    if EMAILJS_SERVICE_ID and EMAILJS_TEMPLATE_ID and EMAILJS_USER_ID and EMAILJS_PRIVATE_KEY and EMAIL_FROM and EMAIL_TO:
        return _send_via_emailjs(subject, body_text or body_html)

    elif SENDGRID_API_KEY and EMAIL_TO and EMAIL_FROM:
        return _send_via_sendgrid(subject, body_html)

    else:
        print("❌ No email provider configured")
        print("   Please set either:")
        print("   - EMAILJS_* variables for EmailJS")
        print("   - SENDGRID_API_KEY for SendGrid")
        return False


def _send_via_emailjs(subject, body_text):
    send_url = 'https://api.emailjs.com/api/v1.0/email/send'
    payload = {
        'service_id': EMAILJS_SERVICE_ID,
        'template_id': EMAILJS_TEMPLATE_ID,
        'user_id': EMAILJS_USER_ID,
        'accessToken': EMAILJS_PRIVATE_KEY,
        'template_params': {
            'to_email': EMAIL_TO,
            'from_email': EMAIL_FROM,
            'subject': subject,
            'message': body_text,
        }
    }

    try:
        r = requests.post(send_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=30)
        if r.status_code == 200:
            print(f'✓ Email sent via EmailJS to {EMAIL_TO}')
            return True
        else:
            print(f'✗ EmailJS send failed: {r.status_code} - {r.text[:200]}')
            return False
    except Exception as e:
        print(f'✗ EmailJS request error: {e}')
        return False


def _send_via_sendgrid(subject, body_html):
    send_url = 'https://api.sendgrid.com/v3/mail/send'
    body = {
        'personalizations': [{'to': [{'email': EMAIL_TO}]}],
        'from': {'email': EMAIL_FROM},
        'subject': subject,
        'content': [{'type': 'text/html', 'value': body_html}]
    }

    try:
        r = requests.post(
            send_url,
            headers={'Authorization': f'Bearer {SENDGRID_API_KEY}', 'Content-Type': 'application/json'},
            json=body,
            timeout=30
        )
        if r.status_code == 202:
            print(f'✓ Email sent via SendGrid to {EMAIL_TO}')
            return True
        else:
            print(f'✗ SendGrid send failed: {r.status_code} - {r.text[:200]}')
            return False
    except Exception as e:
        print(f'✗ SendGrid request error: {e}')
        return False


def build_paper_email(top3_papers):
    subject = "📚 Daily AI Paper Picks"

    def clean_text(text):
        """Remove intro phrases and convert markdown bold to HTML"""
        if not text:
            return text

        # Remove intro text
        text = text.strip()
        for prefix in ["Here is a", "Here's a", "This is a", "Here are", "Summary:", "The summary is:"]:
            if text.lower().startswith(prefix.lower()):
                # Find the first sentence after the colon or newline
                if ':' in text:
                    text = text.split(':', 1)[1].strip()
                elif '\n' in text:
                    text = text.split('\n', 1)[1].strip()
                break

        # Convert markdown **bold** to HTML <strong>
        import re
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)

        return text

    email_body_html = '<h2>Today\'s top 3 papers</h2>'
    for i, t in enumerate(top3_papers, 1):
        email_body_html += f"<h3>{i}. {t['title']}</h3>"
        email_body_html += f"<p><strong>Summary:</strong> {clean_text(t.get('summary', ''))}</p>"
        if t.get('digest'):
            email_body_html += f"<p><strong>Digest:</strong> {clean_text(t['digest'])}</p>"
        if t.get('paper_insights'):
            insights_lines = [line.strip() for line in t['paper_insights'].split('\n') if line.strip()]
            insights_html = '<ul style="margin-top: 5px;">'
            for line in insights_lines:
                if line.startswith('•'):
                    line = line[1:].strip()
                # Convert markdown bold to HTML in insights too
                import re
                line = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', line)
                insights_html += f'<li>{line}</li>'
            insights_html += '</ul>'
            email_body_html += f"<p><strong>Paper Analysis:</strong></p>{insights_html}"
        if t.get('arxiv_link'):
            email_body_html += f"<p><strong>ArXiv:</strong> <a href=\"{t['arxiv_link']}\">{t['arxiv_link']}</a></p>"
        if t.get('serper_found'):
            email_body_html += f"<p><em>Note: Found via Serper search</em></p>"
    email_body_html += '<hr><p><strong>📝 Share your feedback:</strong> Reply to this email with your preferences (e.g., "prefer more papers on X", "less interested in Y") to help refine future picks!</p>'

    email_body_text = 'Today\'s top 3 papers\n\n'
    for i, t in enumerate(top3_papers, 1):
        email_body_text += f"{i}. {t['title']}\n"
        email_body_text += f"Summary: {t.get('summary', '')}\n"
        if t.get('digest'):
            email_body_text += f"Digest: {t['digest']}\n"
        if t.get('paper_insights'):
            email_body_text += f"Paper Analysis:\n{t['paper_insights']}\n"
        if t.get('arxiv_link'):
            email_body_text += f"ArXiv: {t['arxiv_link']}\n"
        if t.get('serper_found'):
            email_body_text += f"Note: Found via Serper search\n"
        email_body_text += '\n'
    email_body_text += '---\n📝 Share your feedback: Reply to this email with your preferences (e.g., "prefer more papers on X", "less interested in Y") to help refine future picks!'

    return subject, email_body_html, email_body_text


if __name__ == "__main__":
    test_papers = [
        {
            'title': 'Test Paper 1',
            'summary': 'This is a test summary',
            'digest': 'Test digest content',
            'paper_insights': '• Key Challenge: Test\n• Methods: Test',
            'arxiv_link': 'https://arxiv.org/abs/1234.5678',
            'serper_found': False
        }
    ]

    subject, html_body, text_body = build_paper_email(test_papers)
    print(f"Subject: {subject}")
    print(f"\nText body:\n{text_body}")
    print("\nAttempting to send test email...")
    send_email(subject, html_body, text_body)
