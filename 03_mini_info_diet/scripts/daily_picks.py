import os, json, requests, random
from dotenv import load_dotenv
load_dotenv()

OLMO_API_KEY = os.environ.get('OLMO_API_KEY')
EMAIL_TO = os.environ.get('EMAIL_TO')
EMAIL_FROM = os.environ.get('EMAIL_FROM')
SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY')
EMAILJS_SERVICE_ID = os.environ.get('EMAILJS_SERVICE_ID')
EMAILJS_TEMPLATE_ID = os.environ.get('EMAILJS_TEMPLATE_ID')
EMAILJS_USER_ID = os.environ.get('EMAILJS_USER_ID')
EMAIL_FROM = os.environ.get('EMAIL_FROM')
EMAIL_TO = os.environ.get('EMAIL_TO')

ROOT = os.path.dirname(os.path.dirname(__file__))
TO_READ = os.path.join(ROOT, 'to_read.csv')
READING = os.path.join(ROOT, 'reading_progress.md')
HISTORY = os.path.join(ROOT, 'picks_history.json')
FEEDBACK = os.path.join(ROOT, 'feedback.json')

# Simple prefilter: read list and exclude already picked
with open(TO_READ) as f:
    titles=[l.strip().split('\t')[-1] for l in f if l.strip()]

picked=set()
if os.path.exists(HISTORY):
    with open(HISTORY) as f:
        hist=json.load(f)
        for entry in hist:
            picked.update(entry.get('titles', []))

candidates=[t for t in titles if t not in picked]
if not candidates:
    candidates=titles

# If many candidates, sample 12 and ask the model to pick top 3
sample = random.sample(candidates, min(len(candidates), 12))

feedback_context = ""
try:
    if os.path.exists(FEEDBACK):
        with open(FEEDBACK) as f:
            feedback_data = json.load(f)
            if feedback_data:
                recent_feedback = feedback_data[-5:]
                prefs = []
                for fb in recent_feedback:
                    if 'preferred_topics' in fb:
                        prefs.extend(fb['preferred_topics'])
                    if 'comment' in fb:
                        prefs.append(fb['comment'])
                if prefs:
                    feedback_context = f"\n\nUser preferences from past feedback: {'; '.join(prefs)}\nConsider these preferences when ranking papers."
except Exception:
    pass

titles_list = '\n'.join(sample)
prompt = f"You are an expert research assistant. From the following list of paper titles (about agentic and multi-agent AI), pick the top 3 most applicable/impactful/innovative/inspiring. Return a JSON object with key 'top3' containing an array of 3 objects with fields: title (string), summary (short 1-sentence summary). Titles:\n{titles_list}{feedback_context}"

resp = requests.post(
    'https://openrouter.ai/api/v1/chat/completions',
    headers={'Authorization': f'Bearer {OLMO_API_KEY}', 'Content-Type': 'application/json'},
    json={'model':'allenai/olmo-3-7b-instruct','messages':[{'role':'user','content':prompt}]}
)

try:
    obj = resp.json()
    content = obj['choices'][0]['message']['content']
    data = json.loads(content)
    top3 = data['top3']
except Exception:
    top3 = [{'title':sample[i],'summary':''} for i in range(min(3,len(sample)))]

# update reading_progress.md
with open(READING) as f:
    readme=f.read()

new_section='\n\nToday\'s top 3 picks:\n\n'
for i, t in enumerate(top3, 1):
    new_section += f"{i}. {t['title']}\n   - Status: not started\n   - Summary: {t.get('summary','')}\n\n"

readme += new_section
with open(READING,'w') as f:
    f.write(readme)

# append history
entry={'date':__import__('datetime').datetime.utcnow().isoformat(),'titles':[t['title'] for t in top3]}
if os.path.exists(HISTORY):
    with open(HISTORY) as f:
        hist=json.load(f)
else:
    hist=[]
hist.append(entry)
with open(HISTORY,'w') as f:
    json.dump(hist,f,indent=2)

# send email via EmailJS if configured
if EMAILJS_SERVICE_ID and EMAILJS_TEMPLATE_ID and EMAILJS_USER_ID and EMAIL_FROM and EMAIL_TO:
    send_url = 'https://api.emailjs.com/api/v1.0/email/send'

    # Build email content with feedback instructions
    email_body = '<h2>Today\'s top 3 papers</h2>' + ''.join([f"<h3>{i}. {t['title']}</h3><p>{t.get('summary','')}</p>" for i, t in enumerate(top3, 1)])
    email_body += '<hr><p><strong>📝 Share your feedback:</strong> Reply to this email with your preferences (e.g., "prefer more papers on X", "less interested in Y") to help refine future picks!</p>'

    payload = {
        'service_id': EMAILJS_SERVICE_ID,
        'template_id': EMAILJS_TEMPLATE_ID,
        'user_id': EMAILJS_USER_ID,
        'template_params': {
            'message_html': email_body,
            'subject': 'Daily paper picks',
            'to_email': EMAIL_TO,
            'from_email': EMAIL_FROM,
        }
    }
    try:
        r = requests.post(send_url, json=payload, headers={'Content-Type':'application/json'})
        if r.status_code == 200:
            print('Email sent via EmailJS')
        else:
            print('EmailJS send failed', r.status_code, r.text)
    except Exception as e:
        print('EmailJS request error', e)
else:
    # fallback to SendGrid if configured
    if SENDGRID_API_KEY and EMAIL_TO and EMAIL_FROM:
        send_url='https://api.sendgrid.com/v3/mail/send'
        body={'personalizations':[{'to':[{'email':EMAIL_TO}]}],'from':{'email':EMAIL_FROM},'subject':'Daily paper picks','content':[{'type':'text/html','value':'<h2>Today\'s top 3 papers</h2>' + ''.join([f"<p><strong>{i}. {t['title']}</strong><br/>{t.get('summary','')}</p>" for i, t in enumerate(top3, 1)])}]}
        requests.post(send_url,headers={'Authorization':f'Bearer {SENDGRID_API_KEY}','Content-Type':'application/json'},json=body)
    else:
        print('No email provider configured; skipping email')
