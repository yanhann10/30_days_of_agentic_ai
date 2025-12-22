import os
import json
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from llm_utils import call_llm

load_dotenv()

SERPER_API_KEY = os.environ.get('SERPER_API_KEY')
OLMO_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OLMO_API_KEY")

def search_arxiv_by_title(title):
    try:
        params = {
            "search_query": f'all:"{title}"',
            "start": 0,
            "max_results": 1
        }
        headers = {"User-Agent": "arxiv-fetcher/0.1"}
        r = requests.get("http://export.arxiv.org/api/query", params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            return None, None

        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None, None

        url_elem = entry.find("atom:id", ns)
        abstract_elem = entry.find("atom:summary", ns)

        url = url_elem.text.strip() if url_elem is not None else None
        abstract = abstract_elem.text.strip() if abstract_elem is not None else None

        return url, abstract
    except Exception as e:
        print(f"Error searching ArXiv: {e}")
        return None, None


def search_serper(title):
    if not SERPER_API_KEY:
        print("Warning: SERPER_API_KEY not set, skipping Serper search")
        return None, None

    try:
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": f"{title} arxiv OR openreview OR paper",
            "num": 10
        })
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers=headers, data=payload, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if 'organic' in data and len(data['organic']) > 0:
                for result in data['organic']:
                    found_url = result.get('link', '')
                    if found_url.startswith('https://openreview.net/pdf/'):
                        snippet = result.get('snippet', '')
                        print(f"  Serper found OpenReview: {found_url}")
                        return found_url, snippet

                for result in data['organic']:
                    found_url = result.get('link', '')
                    if 'arxiv.org' in found_url:
                        snippet = result.get('snippet', '')
                        print(f"  Serper found ArXiv: {found_url}")
                        return found_url, snippet

        print(f"  No OpenReview or ArXiv URL found for: {title}")
        return None, None
    except Exception as e:
        print(f"Error searching Serper: {e}")
        return None, None


def fetch_arxiv_abstract(arxiv_url):
    if not arxiv_url or 'arxiv.org' not in arxiv_url:
        return ''

    try:
        arxiv_id = arxiv_url.split('/abs/')[-1].split('v')[0]
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

        if not OLMO_API_KEY:
            return abstract[:300]

        prompt = f'Summarize this abstract in 1-2 sentences focusing on problem, method, and key result:\n\n{abstract}'
        summary = call_llm(prompt, max_retries=2)

        if summary:
            return summary
        else:
            return abstract[:300] if abstract else ''
    except Exception as e:
        print(f"Error fetching ArXiv abstract: {e}")
        return ''


def enrich_paper_info(title, arxiv_link_from_csv=''):
    result = {
        'arxiv_link': arxiv_link_from_csv,
        'digest': '',
        'serper_found': False
    }

    if not result['arxiv_link']:
        print(f"Searching ArXiv for: {title}")
        arxiv_url, abstract = search_arxiv_by_title(title)
        if arxiv_url:
            result['arxiv_link'] = arxiv_url
            result['digest'] = abstract[:300] if abstract else ''
            print(f"  Found ArXiv: {arxiv_url}")
        else:
            print(f"  ArXiv not found, searching Serper...")
            serper_url, snippet = search_serper(title)
            if serper_url:
                result['arxiv_link'] = serper_url
                result['digest'] = snippet if snippet else ''
                result['serper_found'] = True
                print(f"  Serper found: {serper_url}")
            else:
                print(f"  No results found for: {title}")

    if result['arxiv_link'] and 'arxiv.org' in result['arxiv_link'] and not result['digest']:
        result['digest'] = fetch_arxiv_abstract(result['arxiv_link'])

    return result


if __name__ == "__main__":
    test_title = "ToolRL: Reward is All Tool Learning Needs"
    print(f"Testing with: {test_title}\n")

    info = enrich_paper_info(test_title)
    print("\nResults:")
    print(f"ArXiv Link: {info['arxiv_link']}")
    print(f"Digest: {info['digest'][:100]}..." if info['digest'] else "Digest: (none)")
    print(f"Found via Serper: {info['serper_found']}")
