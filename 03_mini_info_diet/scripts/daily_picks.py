import os, json, requests, random, xml.etree.ElementTree as ET
from dotenv import load_dotenv
load_dotenv()

def fetch_arxiv_abstract(arxiv_url):
    if not arxiv_url:
        return ''
    try:
        arxiv_id = arxiv_url.split('/abs/')[-1]
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        r = requests.get(api_url, timeout=10)
        if r.status_code != 200:
            return ''
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return ''
        abstract_elem = entry.find("atom:summary", ns)
        if abstract_elem is None:
            return ''
        abstract = abstract_elem.text.strip()
        digest_resp = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={'Authorization': f'Bearer {os.environ.get("OLMO_API_KEY")}', 'Content-Type': 'application/json'},
            json={'model':'allenai/olmo-3-7b-instruct','messages':[{'role':'user','content':f'Summarize this abstract in 1-2 sentences focusing on problem, method, and key result:\n\n{abstract}'}]},
            timeout=15
        )
        if digest_resp.status_code == 200:
            digest_obj = digest_resp.json()
            return digest_obj['choices'][0]['message']['content'].strip()
        return abstract[:300]
    except Exception:
        return ''

OLMO_API_KEY = os.environ.get('OLMO_API_KEY')
EMAIL_TO = os.environ.get('EMAIL_TO')
EMAIL_FROM = os.environ.get('EMAIL_FROM')
SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY')
EMAILJS_SERVICE_ID = os.environ.get('EMAILJS_SERVICE_ID')
EMAILJS_TEMPLATE_ID = os.environ.get('EMAILJS_TEMPLATE_ID')
EMAILJS_USER_ID = os.environ.get('EMAILJS_USER_ID')
EMAILJS_PRIVATE_KEY = os.environ.get('EMAILJS_PRIVATE_KEY')
EMAIL_FROM = os.environ.get('EMAIL_FROM')
EMAIL_TO = os.environ.get('EMAIL_TO')

ROOT = os.path.dirname(os.path.dirname(__file__))
TO_READ = os.path.join(ROOT, 'to_read.csv')
READING = os.path.join(ROOT, 'reading_progress.md')
HISTORY = os.path.join(ROOT, 'picks_history.json')
FEEDBACK = os.path.join(ROOT, 'feedback.json')

papers = []
with open(TO_READ) as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        paper = {
            'title': parts[0],
            'arxiv_link': parts[1] if len(parts) > 1 else '',
            'digest': ''
        }
        papers.append(paper)

titles = [p['title'] for p in papers]

picked=set()
if os.path.exists(HISTORY):
    with open(HISTORY) as f:
        hist=json.load(f)
        for entry in hist:
            picked.update(entry.get('titles', []))

candidates=[p for p in papers if p['title'] not in picked]
if not candidates:
    candidates=papers

# If many candidates, sample 12 and ask the model to pick top 3
sample = random.sample(candidates, min(len(candidates), 12))
sample_titles = [p['title'] for p in sample]

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

titles_list = '\n'.join(sample_titles)
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
    top3 = [{'title':sample[i]['title'],'summary':''} for i in range(min(3,len(sample)))]

papers_dict = {p['title']: p for p in papers}
for item in top3:
    if item['title'] in papers_dict:
        item['arxiv_link'] = papers_dict[item['title']]['arxiv_link']
        item['digest'] = fetch_arxiv_abstract(item['arxiv_link'])
    else:
        item['digest'] = ''
        item['arxiv_link'] = ''

# update reading_progress.md
with open(READING) as f:
    readme=f.read()

new_section='\n\nToday\'s top 3 picks:\n\n'
for i, t in enumerate(top3, 1):
    new_section += f"{i}. {t['title']}\n   - Status: not started\n   - Summary: {t.get('summary','')}\n"
    if t.get('digest'):
        new_section += f"   - Digest: {t['digest']}\n"
    if t.get('arxiv_link'):
        new_section += f"   - ArXiv: {t['arxiv_link']}\n"
    new_section += '\n'

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

email_body_html = '<h2>Today\'s top 3 papers</h2>'
for i, t in enumerate(top3, 1):
    email_body_html += f"<h3>{i}. {t['title']}</h3>"
    email_body_html += f"<p><strong>Summary:</strong> {t.get('summary','')}</p>"
    if t.get('digest'):
        email_body_html += f"<p><strong>Digest:</strong> {t['digest']}</p>"
    if t.get('arxiv_link'):
        email_body_html += f"<p><strong>ArXiv:</strong> <a href=\"{t['arxiv_link']}\">{t['arxiv_link']}</a></p>"
email_body_html += '<hr><p><strong>📝 Share your feedback:</strong> Reply to this email with your preferences (e.g., "prefer more papers on X", "less interested in Y") to help refine future picks!</p>'

email_body_text = 'Today\'s top 3 papers\n\n'
for i, t in enumerate(top3, 1):
    email_body_text += f"{i}. {t['title']}\n"
    email_body_text += f"Summary: {t.get('summary','')}\n"
    if t.get('digest'):
        email_body_text += f"Digest: {t['digest']}\n"
    if t.get('arxiv_link'):
        email_body_text += f"ArXiv: {t['arxiv_link']}\n"
    email_body_text += '\n'
email_body_text += '---\n📝 Share your feedback: Reply to this email with your preferences (e.g., "prefer more papers on X", "less interested in Y") to help refine future picks!'

if EMAILJS_SERVICE_ID and EMAILJS_TEMPLATE_ID and EMAILJS_USER_ID and EMAILJS_PRIVATE_KEY and EMAIL_FROM and EMAIL_TO:
    send_url = 'https://api.emailjs.com/api/v1.0/email/send'
    payload = {
        'service_id': EMAILJS_SERVICE_ID,
        'template_id': EMAILJS_TEMPLATE_ID,
        'user_id': EMAILJS_USER_ID,
        'accessToken': EMAILJS_PRIVATE_KEY,
        'template_params': {
            'to_email': EMAIL_TO,
            'from_email': EMAIL_FROM,
            'subject': 'Daily paper picks',
            'message': email_body_text,
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
elif SENDGRID_API_KEY and EMAIL_TO and EMAIL_FROM:
    send_url='https://api.sendgrid.com/v3/mail/send'
    body={'personalizations':[{'to':[{'email':EMAIL_TO}]}],'from':{'email':EMAIL_FROM},'subject':'Daily paper picks','content':[{'type':'text/html','value':email_body_html}]}
    try:
        r = requests.post(send_url,headers={'Authorization':f'Bearer {SENDGRID_API_KEY}','Content-Type':'application/json'},json=body)
        if r.status_code == 202:
            print('Email sent via SendGrid')
        else:
            print('SendGrid send failed', r.status_code, r.text)
    except Exception as e:
        print('SendGrid request error', e)
else:
    print('No email provider configured; skipping email')
