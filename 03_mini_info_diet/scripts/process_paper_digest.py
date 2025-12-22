
import os
import re
import json
import requests
import tempfile
from dotenv import load_dotenv
import pymupdf
from llm_utils import call_llm


load_dotenv()

ROOT = os.path.dirname(os.path.dirname(__file__))
READING = os.path.join(ROOT, 'reading_progress.md')
OLMO_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OLMO_API_KEY")


def extract_latest_picks(reading_file):
    try:
        with open(reading_file, 'r', encoding='utf-8') as f:
            content = f.read()

        sections = content.split("Today's top 3 picks:")
        if len(sections) < 2:
            return []

        latest_section = sections[-1]
        picks = []

        for i in range(1, 4):
            title_pattern = rf'{i}\.\s+(.+?)\n'
            title_match = re.search(title_pattern, latest_section)
            if not title_match:
                continue

            title = title_match.group(1).strip()

            arxiv_pattern = rf'{i}\.\s+.*?\n(?:\s+-[^\n]+\n)*?\s+- ArXiv:\s*(https?://[^\s\n]+)'
            arxiv_match = re.search(arxiv_pattern, latest_section, re.DOTALL)
            arxiv_url = arxiv_match.group(1).strip() if arxiv_match else None

            picks.append({'title': title, 'arxiv_url': arxiv_url, 'index': i})

        return picks
    except Exception as e:
        print(f"Error extracting picks: {e}")
        return []


def download_arxiv_pdf(arxiv_url):
    try:
        arxiv_id = arxiv_url.split('/abs/')[-1].split('v')[0]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        response = requests.get(pdf_url, timeout=30, stream=True)
        if response.status_code != 200:
            print(f"Failed to download PDF from {pdf_url}: {response.status_code}")
            return None

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()

        return temp_file.name
    except Exception as e:
        print(f"Error downloading PDF from {arxiv_url}: {e}")
        return None


def extract_pdf_text(pdf_path):
    try:
        doc = pymupdf.open(pdf_path)
        text_parts = []

        for page_num, page in enumerate(doc):
            if page_num >= 50:
                break
            text = page.get_text()
            if text.strip():
                text_parts.append(text)

        doc.close()
        full_text = '\n\n'.join(text_parts)

        if len(full_text) > 100000:
            full_text = full_text[:100000] + "\n\n[Text truncated...]"

        return full_text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None


def extract_paper_insights(text, title):
    if not OLMO_API_KEY:
        print("OLMO_API_KEY not found, skipping LLM extraction")
        return None

    prompt = f"""Extract key insights from this research paper titled "{title}".

Paper text (excerpt):
{text[:50000]}

Provide EXACTLY 4 bullet points in this format (one bullet per line, start each with "• "):

• Key Challenge: [main problem in 1-2 sentences]
• Methods: [key techniques in 1-2 sentences]
• Innovation: [novel contributions in 1-2 sentences]
• Limitations: [constraints in 1-2 sentences]

Be specific and technical. Each bullet must start with "• " followed by the category name."""

    result = call_llm(prompt)
    return result


def update_reading_progress(reading_file, pick_index, insights):
    try:
        with open(reading_file, 'r', encoding='utf-8') as f:
            content = f.read()

        sections = content.rsplit("Today's top 3 picks:", 1)
        if len(sections) < 2:
            print("Could not find 'Today's top 3 picks' section")
            return False

        latest_section = sections[1]

        pattern = rf'({pick_index}\.\s+[^\n]+(?:\n\s+-[^\n]+)*?\n\s+- ArXiv:\s*[^\n]+)'
        match = re.search(pattern, latest_section, re.MULTILINE)

        if match:
            if 'Key Challenge:' in latest_section or '• Key Challenge:' in latest_section:
                print(f"Insights already added for pick {pick_index}, skipping")
                return True

            replacement = match.group(1) + f'\n   - Paper Analysis:\n'
            for line in insights.split('\n'):
                if line.strip():
                    line = line.strip()
                    if line.startswith('•'):
                        line = '     ' + line
                    elif not line.startswith(' '):
                        line = '     • ' + line
                    replacement += f'   {line}\n'

            updated_section = latest_section.replace(match.group(1), replacement)
            updated_content = sections[0] + "Today's top 3 picks:" + updated_section

            with open(reading_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)

            return True
        else:
            print(f"Could not find pick {pick_index} in latest section")
            return False
    except Exception as e:
        print(f"Error updating reading_progress.md: {e}")
        return False


def process_paper(pick):
    print(f"\nProcessing: {pick['title']}")

    if not pick.get('arxiv_url'):
        print(f"No ArXiv URL found for {pick['title']}, skipping PDF processing")
        return False

    print(f"ArXiv URL: {pick['arxiv_url']}")

    pdf_path = download_arxiv_pdf(pick['arxiv_url'])
    if not pdf_path:
        print(f"Failed to download PDF for {pick['title']}")
        return False

    try:
        print("Extracting text from PDF...")
        text = extract_pdf_text(pdf_path)
        if not text:
            print(f"Failed to extract text from PDF")
            return False

        print(f"Extracted {len(text)} characters from PDF")

        print("Calling LLM to extract insights...")
        insights = extract_paper_insights(text, pick['title'])
        if not insights:
            print(f"Failed to get insights from LLM")
            return False

        print("Insights extracted successfully")

        print("Updating reading_progress.md...")
        success = update_reading_progress(READING, pick['index'], insights)
        if success:
            print(f"Successfully updated reading_progress.md for {pick['title']}")
        else:
            print(f"Failed to update reading_progress.md")

        return success
    finally:
        try:
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)
        except Exception as e:
            print(f"Warning: Could not delete temp file {pdf_path}: {e}")


def main():
    print("Starting paper digest processing...")

    if not OLMO_API_KEY:
        print("Warning: OLMO_API_KEY not set. LLM extraction will be skipped.")

    picks = extract_latest_picks(READING)
    if not picks:
        print("No picks found in reading_progress.md")
        return

    print(f"Found {len(picks)} papers to process")

    for pick in picks:
        try:
            process_paper(pick)
        except Exception as e:
            print(f"Error processing {pick['title']}: {e}")
            print("Continuing with next paper...")
            continue

    print("\nPaper digest processing complete!")


if __name__ == "__main__":
    main()

