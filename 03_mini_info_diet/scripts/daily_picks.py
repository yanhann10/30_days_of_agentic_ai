import os, json, requests, random, re, datetime
from dotenv import load_dotenv

load_dotenv()

from search_paper_info import enrich_paper_info
from process_paper_digest import download_arxiv_pdf, extract_pdf_text, extract_paper_insights
from llm_utils import call_llm_json
from send_email import send_email, build_paper_email

def extract_paper_analysis(reading_file, title):
    try:
        with open(reading_file, 'r', encoding='utf-8') as f:
            content = f.read()

        sections = content.rsplit("Today's top 3 picks:", 1)
        if len(sections) < 2:
            return None

        latest_section = sections[1]

        title_escaped = re.escape(title)
        pattern = rf'\d+\.\s+{title_escaped}.*?(\n\s+- Paper Analysis:\n(?:\s+[^\n]+\n)+)'
        match = re.search(pattern, latest_section, re.DOTALL | re.IGNORECASE)

        if not match:
            title_words = title.split()
            if len(title_words) > 3:
                partial_title = ' '.join(title_words[:3])
                partial_escaped = re.escape(partial_title)
                pattern = rf'\d+\.\s+.*?{partial_escaped}.*?(\n\s+- Paper Analysis:\n(?:\s+[^\n]+\n)+)'
                match = re.search(pattern, latest_section, re.DOTALL | re.IGNORECASE)

        if match:
            analysis_text = match.group(1).strip()
            lines = analysis_text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('- Paper Analysis:'):
                    line = re.sub(r'^\s*[-•]\s*', '', line)
                    if line:
                        cleaned_lines.append(line)
            return '\n'.join(cleaned_lines) if cleaned_lines else None

        return None
    except Exception as e:
        print(f"Error extracting paper analysis: {e}")
        return None

ROOT = os.path.dirname(os.path.dirname(__file__))
TO_READ = os.path.join(ROOT, 'to_read.csv')
READING = os.path.join(ROOT, 'reading_progress.md')
HISTORY = os.path.join(ROOT, 'picks_history.json')
FEEDBACK = os.path.join(ROOT, 'feedback.json')
PREFS_JSONL = os.path.join(ROOT, 'prefs.jsonl')

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

sample = random.sample(candidates, min(len(candidates), 12))
sample_titles = [p['title'] for p in sample]

feedback_context = ""
try:
    prefs_avoid = []
    prefs_more = []
    if os.path.exists(PREFS_JSONL) and os.path.getsize(PREFS_JSONL) > 0:
        with open(PREFS_JSONL, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for ln in lines[-50:]:
            try:
                ev = json.loads(ln)
            except Exception:
                continue
            prefs_avoid.extend(ev.get("topics_to_avoid") or [])
            prefs_more.extend(ev.get("topics_to_increase") or [])
    def _uniq(xs):
        out, seen = [], set()
        for x in xs:
            x = str(x).strip()
            if not x:
                continue
            k = x.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out
    prefs_avoid = _uniq(prefs_avoid)[:12]
    prefs_more = _uniq(prefs_more)[:12]
    if prefs_more or prefs_avoid:
        chunks = []
        if prefs_more:
            chunks.append("Read more of: " + "; ".join(prefs_more))
        if prefs_avoid:
            chunks.append("Read less of: " + "; ".join(prefs_avoid))
        feedback_context = "\n\nUser preferences from email feedback:\n" + "\n".join(chunks) + "\nConsider these preferences when ranking papers."
except Exception:
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
                    feedback_context = feedback_context + f"\n\nUser preferences from past feedback: {'; '.join(prefs)}\nConsider these preferences when ranking papers."
except Exception:
    pass

titles_list = '\n'.join(sample_titles)
prompt = f"You are an expert research assistant. From the following list of paper titles (about agentic and multi-agent AI), pick the top 3 most applicable/impactful/innovative/inspiring. Return a JSON object with key 'top3' containing an array of 3 objects with fields: title (string), summary (short 1-sentence summary). Titles:\n{titles_list}{feedback_context}"

print("Selecting top 3 papers using LLM...")
data = call_llm_json(prompt)

if data and 'top3' in data:
    top3 = data['top3']
else:
    print("LLM failed to pick papers, using random selection")
    top3 = [{'title':sample[i]['title'],'summary':''} for i in range(min(3,len(sample)))]

if len(top3) < 3:
    fallback = [p for p in sample if p['title'] not in {t.get('title') for t in top3}]
    while len(top3) < 3 and fallback:
        p = fallback.pop(0)
        top3.append({'title': p['title'], 'summary': ''})
    if len(top3) < 3:
        while len(top3) < 3 and len(papers) > len(top3):
            top3.append({'title': papers[len(top3)]['title'], 'summary': ''})

papers_dict = {p['title']: p for p in papers}
papers_dict_normalized = {p['title'].lower().strip(): p for p in papers}

for item in top3:
    title = item['title']
    item['paper_insights'] = None

    arxiv_link_from_csv = ''
    if title in papers_dict:
        arxiv_link_from_csv = papers_dict[title]['arxiv_link']
    else:
        title_normalized = title.lower().strip()
        if title_normalized in papers_dict_normalized:
            arxiv_link_from_csv = papers_dict_normalized[title_normalized]['arxiv_link']

    print(f"\nEnriching info for: {title}")
    paper_info = enrich_paper_info(title, arxiv_link_from_csv)

    item['arxiv_link'] = paper_info['arxiv_link']
    item['digest'] = paper_info['digest']
    if paper_info['serper_found']:
        item['serper_found'] = True
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OLMO_API_KEY")

    if item.get('arxiv_link') and 'arxiv.org' in item['arxiv_link'] and OPENROUTER_API_KEY:
        print(f"Downloading PDF and extracting insights for: {title}")
        pdf_path = download_arxiv_pdf(item['arxiv_link'])
        if pdf_path:
            try:
                text = extract_pdf_text(pdf_path)
                if text:
                    insights = extract_paper_insights(text, title)
                    if insights:
                        item['paper_insights'] = insights
                        print(f"  Insights extracted successfully")
                try:
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                except:
                    pass
            except Exception as e:
                print(f"  Error extracting insights: {e}")

total_papers = len(papers)
papers_sent = len(picked)
progress_pct = (papers_sent / total_papers * 100) if total_papers > 0 else 0

with open(READING) as f:
    readme=f.read()

if readme.startswith('# Reading progress'):
    lines = readme.split('\n')
    if len(lines) > 2 and lines[1].strip() == '' and lines[2].startswith('Progress:'):
        lines[2] = f'Progress: {papers_sent}/{total_papers} papers sent ({progress_pct:.1f}%)'
        readme = '\n'.join(lines)
    else:
        lines.insert(2, f'Progress: {papers_sent}/{total_papers} papers sent ({progress_pct:.1f}%)')
        readme = '\n'.join(lines)
else:
    readme = f'# Reading progress\n\nProgress: {papers_sent}/{total_papers} papers sent ({progress_pct:.1f}%)\n\n' + readme

today_date = datetime.datetime.now().strftime('%Y-%m-%d')

new_section=f'\n\n{today_date}\n'
for i, t in enumerate(top3, 1):
    new_section += f"{i} | {t['title']} | 0\n"

readme += new_section
with open(READING,'w') as f:
    f.write(readme)

entry={'date':__import__('datetime').datetime.utcnow().isoformat(),'titles':[t['title'] for t in top3]}
if os.path.exists(HISTORY):
    with open(HISTORY) as f:
        hist=json.load(f)
else:
    hist=[]
hist.append(entry)
with open(HISTORY,'w') as f:
    json.dump(hist,f,indent=2)

print("\nPreparing email...")
print("\nDEBUG: Top 3 papers data:")
for i, paper in enumerate(top3, 1):
    print(f"\nPaper {i}: {paper.get('title', 'N/A')}")
    print(f"  - Has digest: {bool(paper.get('digest'))}")
    print(f"  - Digest length: {len(paper.get('digest', ''))} chars")
    print(f"  - Has paper_insights: {bool(paper.get('paper_insights'))}")
    if paper.get('paper_insights'):
        print(f"  - Insights length: {len(paper['paper_insights'])} chars")
        print(f"  - Insights preview: {paper['paper_insights'][:200]}...")

subject, email_body_html, email_body_text = build_paper_email(top3)

# Print email content for debugging
print("\n" + "="*80)
print("EMAIL SUBJECT:", subject)
print("="*80)
print("\nEMAIL TEXT BODY:")
print(email_body_text)
print("\n" + "="*80)
print("\nEMAIL HTML BODY (full):")
print(email_body_html)
print("="*80)

# Uncomment to send email
# send_email(subject, email_body_html, email_body_text)
