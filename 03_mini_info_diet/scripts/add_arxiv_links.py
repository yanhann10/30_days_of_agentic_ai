import os
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
load_dotenv()
import xml.etree.ElementTree as ET
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
SERPER_KEY = os.environ.get('SERPER_API_KEY')
TEST_LIMIT = int(os.environ.get('TEST_LIMIT','0'))  # 0 means no limit
TO_READ = os.path.join(ROOT, 'to_read.csv')
TMP = os.path.join(ROOT, 'to_read_with_links.csv')

with open(TO_READ, 'r') as f:
    lines = [l.rstrip('\n') for l in f if l.strip()]

out_lines = []
count = 0

for line in lines:
    parts = line.split('\t')
    title = parts[-1].strip()
    existing = parts[1].strip() if len(parts) > 1 else ''
    if existing.startswith('http'):
        out_lines.append(f"{title}\t{existing}")
        print(f"Already has link: {title}")
        continue

    
    if TEST_LIMIT and count >= TEST_LIMIT:
        out_lines.append(f"{title}\t")
        count += 1
        continue

    url = ''
    # Try Serper first if available
    if SERPER_KEY:
        try:
            headers = {'X-API-KEY': SERPER_KEY}
            params = {'q': title, 'num': 5}
            r = requests.get('https://google.serper.dev/search', headers=headers, params=params, timeout=10)
            if r.status_code == 200:
                j = r.json()
                organic = j.get('organic', [])
                for item in organic:
                    link = item.get('link') or item.get('url')
                    if link and 'arxiv.org' in link:
                        url = link
                        break
                if not url:
                    for k in ('top', 'knowledgeGraph'):
                        sec = j.get(k, {})
                        if isinstance(sec, dict):
                            link = sec.get('url') or sec.get('link')
                            if link and 'arxiv.org' in link:
                                url = link
                                break
            else:
                print(f"Serper returned {r.status_code} for '{title}'")
        except Exception as e:
            print(f"Serper error for '{title}': {e}")

    # Fallback to arXiv API if no url found
    if not url:
        try:
            query = f'all:"{title}"'
            params = {'search_query': query, 'start': 0, 'max_results': 1}
            r = requests.get('http://export.arxiv.org/api/query', params=params, timeout=10)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = root.find('atom:entry', ns)
            if entry is not None:
                aid = entry.find('atom:id', ns)
                if aid is not None and aid.text:
                    url = aid.text.strip()
        except Exception as e:
            print(f"arXiv error for '{title}': {e}")

    out_lines.append(f"{title}\t{url}")
    if url:
        print(f"Found: {title} -> {url}")
    else:
        print(f"No link for: {title}")

    count += 1


with open(TMP, 'w') as f:
    f.write('\n'.join(out_lines) + '\n')

# replace original file
os.replace(TMP, TO_READ)
print('Updated to_read.csv with arXiv links where found.')
