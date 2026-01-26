# Self-Evolving Agents KG - Architecture

## Pipeline Overview

```
GitHub README --> Paper Metadata --> PDFs --> Parsed Text --> Methods --> Visualization
      |                |              |           |              |             |
  parse_awesome   papers.json    download     parsed_      finegrained_    viz/index.html
   _readme.py                    _papers.py   papers.json   methods.json
```

## Components

### 1. Data Collection
**`src/parse_awesome_readme.py`**
- Parses Awesome-Self-Evolving-Agents GitHub README
- Extracts: title, venue, year, arXiv ID, subcategory
- Output: `data/processed/papers.json` (200 papers)

**`src/download_papers.py`**
- Downloads PDFs from arXiv (rate-limited)
- Selects N papers per subcategory for balanced coverage
- Output: `data/pdfs/*.pdf`

### 2. Text Extraction
**`src/parse_papers.py`**
- Uses PyMuPDF (free) or LlamaParse (optional)
- Extracts full text and abstract
- Output: `data/processed/parsed_papers.json`

### 3. Method Extraction
**`src/extract_finegrained.py`**
- Categories from GitHub repo section subheaders
- Domain-specific methods with actual technique subnodes
- Extracts context sentences showing how methods enable evolution
- Output: `data/processed/finegrained_methods.json`

### 4. Visualization
**`src/visualize_finegrained.py`**
- D3.js force-directed graph
- Click node to see papers
- Shows "how it works" for each method
- Year-based animation play button
- Deployable as static HTML to Vercel
- Output: `viz/index.html`

## Data Flow

```
papers.json (200 papers)
    |
    v  select by subcategory
downloaded_papers.json (112 papers)
    |
    v  PDF parsing / title matching
parsed_papers.json (or fallback to papers.json)
    |
    v  method extraction
finegrained_methods.json (38 methods)
finegrained_connections.json (19 connections)
    |
    v  visualization
viz/index.html (interactive graph)
```

## Usage

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install pymupdf httpx

# Run pipeline
python src/parse_awesome_readme.py
python src/download_papers.py --limit 200
python src/parse_papers.py
python src/extract_finegrained.py
python src/visualize_finegrained.py

# View
open viz/index.html
```

---

AI generated and human reviewed
