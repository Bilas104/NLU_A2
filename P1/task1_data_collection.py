"""
=============================================================================
TASK 1: DATASET PREPARATION
=============================================================================
This script collects textual data from IIT Jodhpur sources, preprocesses it,
and generates corpus statistics + a word cloud.

Sources used (>=3 required):
  1. IIT Jodhpur official website (department pages, about, programs)
  2. Academic regulation PDF (mandatory as per assignment)
  3. Academic Programs

Output: cleaned_corpus.txt, corpus_stats.txt, wordcloud.png
=============================================================================
"""

import requests
from bs4 import BeautifulSoup
import re
import os
import string
import time
import tempfile
from urllib.parse import urljoin, urlparse
import nltk
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

URLS = [
    # Main website pages
    "https://iitj.ac.in/",

    # Academic Regulations
    "https://iitj.ac.in/office-of-academics/en/academic-regulations",

    # Department pages
    "https://iitj.ac.in/computer-science-engineering/",
    "https://iitj.ac.in/electrical-engineering/",
    "https://iitj.ac.in/mathematics/",
    "https://iitj.ac.in/physics/",
    "https://iitj.ac.in/chemistry/",
    "https://iitj.ac.in/bioscience-bioengineering/",
    "https://iitj.ac.in/civil-and-infrastructure-engineering/",

    # Academic Programs
    "https://www.iitj.ac.in/computer-science-engineering/en/postgraduate-programs",
    "https://iitj.ac.in/mathematics/en/postgraduate-programs",
]

# Request settings
REQUEST_TIMEOUT = 20       # seconds per request
CRAWL_DELAY = 0.5          # seconds between requests
MAX_PAGES = 500            # safety cap
MAX_PDF_SIZE_MB = 50       # skip PDFs larger than this
 
# HTTP headers to mimic a real browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

# ============================================================================
# STEP 1: URL UTILITIES
# ============================================================================
 
def is_iitj_domain(url):
    """
    Checks if a URL belongs to *.iitj.ac.in (any subdomain).
    This ensures we never crawl outside IIT Jodhpur's web presence.
    Examples:
        https://iitj.ac.in/about       → True
        https://cse.iitj.ac.in/        → True
        https://google.com              → False
    """
    try:
        hostname = urlparse(url).hostname or ""
        return hostname.endswith("iitj.ac.in")
    except Exception:
        return False
 
 
def is_pdf_url(url):
    """
    Checks if a URL points to a PDF file based on its extension.
    Also handles URLs with query parameters like file.pdf?v=2
    """
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")
 
 
def normalize_url(url):
    """
    Normalizes a URL by removing fragments (#section) and trailing slashes.
    This prevents visiting the same page multiple times due to minor URL differences.
    """
    parsed = urlparse(url)
    # Remove fragment, keep everything else
    clean = parsed._replace(fragment="")
    normalized = clean.geturl().rstrip("/")
    return normalized
 
 
def is_useful_link(url):
    """
    Filters out non-content links: images, stylesheets, scripts, etc.
    We only want HTML pages and PDFs.
    """
    path = urlparse(url).path.lower()
 
    # Skip common non-content file extensions
    skip_extensions = [
        '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.ico',  # images
        '.css', '.js', '.json', '.xml',                             # code/data
        '.zip', '.tar', '.gz', '.rar',                              # archives
        '.mp3', '.mp4', '.avi', '.mov',                             # media
        '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',         # office (non-pdf)
        '.woff', '.woff2', '.ttf', '.eot',                         # fonts
    ]
    for ext in skip_extensions:
        if path.endswith(ext):
            return False
 
    return True
 
 
# ============================================================================
# STEP 2: WEB PAGE SCRAPING (with table extraction)
# ============================================================================
 
def extract_tables(soup):
    """
    Extracts text content from HTML <table> elements.
    Converts each table into readable rows of text, preserving structure.
 
    This is important because many IIT Jodhpur pages use tables for:
    - Course listings (course code, name, credits)
    - Faculty info (name, designation, research area)
    - Academic calendars, fee structures, etc.
 
    Returns a list of strings, each representing one table's content.
    """
    table_texts = []
 
    for table in soup.find_all("table"):
        rows_text = []
 
        for row in table.find_all("tr"):
            # Extract text from each cell (th or td)
            cells = row.find_all(["th", "td"])
            cell_texts = []
            for cell in cells:
                # Get cell text, collapse whitespace
                text = cell.get_text(separator=" ", strip=True)
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    cell_texts.append(text)
 
            if cell_texts:
                # Join cells with " | " separator for readability
                row_text = " | ".join(cell_texts)
                rows_text.append(row_text)
 
        if rows_text:
            # Join all rows with newline → one block per table
            table_text = "\n".join(rows_text)
            table_texts.append(table_text)
 
    return table_texts
 
 
def scrape_webpage(url):
    """
    Fetches a webpage and extracts:
      1. Main body text (paragraphs, headings, lists)
      2. Tabular data from <table> elements
      3. All links found on the page (for crawling)
 
    Returns: (page_text, list_of_links)
    """
    try:
        response = requests.get(
            url, headers=HEADERS, timeout=REQUEST_TIMEOUT, verify=False
        )
        response.raise_for_status()
 
        # Skip non-HTML responses
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type.lower():
            return "", []
 
        soup = BeautifulSoup(response.text, "html.parser")
 
        # ---- Extract links BEFORE removing nav/footer ----
        # We want to discover links from the full page
        discovered_links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            # Resolve relative URLs to absolute
            full_url = urljoin(url, href)
            full_url = normalize_url(full_url)
 
            # Only keep iitj.ac.in links that look useful
            if is_iitj_domain(full_url) and is_useful_link(full_url):
                discovered_links.append(full_url)
 
        # ---- Extract tabular data ----
        table_texts = extract_tables(soup)
 
        # ---- Extract main text ----
        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "noscript", "iframe", "meta", "link", "form"]):
            tag.decompose()
 
        # Get body text
        body_text = soup.get_text(separator=" ", strip=True)
 
        # Combine body text + table text
        all_text_parts = [body_text]
        all_text_parts.extend(table_texts)
        combined_text = "\n\n".join(all_text_parts)
 
        return combined_text, discovered_links
 
    except Exception as e:
        print(f"    [WARN] Failed: {url} → {e}")
        return "", []
 
 
# ============================================================================
# STEP 3: PDF DOWNLOADING AND TEXT EXTRACTION
# ============================================================================
 
def download_and_extract_pdf(url):
    """
    Downloads a PDF from a URL into a temp file, extracts text using PyPDF2.
    Handles large PDFs gracefully by checking Content-Length first.
    Also extracts text from tables embedded in the PDF.
 
    Returns: extracted text as a string (empty on failure)
    """
    try:
        # First, check file size with a HEAD request
        head = requests.head(url, headers=HEADERS, timeout=10, verify=False)
        content_length = int(head.headers.get("Content-Length", 0))
        if content_length > MAX_PDF_SIZE_MB * 1024 * 1024:
            print(f"    [SKIP] PDF too large ({content_length // (1024*1024)}MB): {url}")
            return ""
 
        # Download the PDF
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, verify=False)
        response.raise_for_status()
 
        # Save to a temporary file and extract text
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
 
        from PyPDF2 import PdfReader
        reader = PdfReader(tmp_path)
 
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
 
        # Clean up temp file
        os.unlink(tmp_path)
 
        full_text = "\n".join(text_parts)
        if full_text.strip():
            print(f"    [PDF] Extracted {len(reader.pages)} pages from {url}")
        return full_text
 
    except Exception as e:
        print(f"    [WARN] PDF failed: {url} → {e}")
        # Clean up temp file if it exists
        try:
            os.unlink(tmp_path)
        except:
            pass
        return ""
 
 
# ============================================================================
# STEP 4: TWO-LEVEL CRAWLER
# ============================================================================
 
def crawl(seed_urls):
    """
    Two-level crawler:
      Level 1: Visit each seed URL, extract text + discover links
      Level 2: Visit each discovered link (if within *.iitj.ac.in)
 
    Keeps track of visited URLs to avoid duplicates.
    Separates HTML pages and PDFs for appropriate handling.
 
    Returns:
      - documents: list of (source_url, extracted_text) tuples
    """
    visited = set()          # URLs we've already processed
    documents = []           # collected (url, text) pairs
    level2_urls = set()      # links discovered from seed pages
 
    total_pages = 0
 
    # ---- LEVEL 1: Seed pages ----
    print(f"\n  === LEVEL 1: Crawling {len(seed_urls)} seed pages ===\n")
 
    for i, url in enumerate(seed_urls):
        url = normalize_url(url)
        if url in visited:
            continue
        visited.add(url)
 
        print(f"  [{i+1}/{len(seed_urls)}] {url}")
 
        if is_pdf_url(url):
            # Seed URL is a PDF — download and extract
            text = download_and_extract_pdf(url)
            if text.strip():
                documents.append((url, text))
        else:
            # Seed URL is an HTML page — scrape text + discover links
            text, links = scrape_webpage(url)
            if text.strip():
                documents.append((url, text))
 
            # Collect discovered links for level 2
            for link in links:
                if link not in visited:
                    level2_urls.add(link)
 
        total_pages += 1
        time.sleep(CRAWL_DELAY)  # be polite
 
    # ---- LEVEL 2: Follow discovered links ----
    level2_list = list(level2_urls)
    print(f"\n  === LEVEL 2: Found {len(level2_list)} links to follow ===\n")
 
    for i, url in enumerate(level2_list):
        # Safety cap
        if total_pages >= MAX_PAGES:
            print(f"\n  [STOP] Reached max page limit ({MAX_PAGES}). Stopping.")
            break
 
        if url in visited:
            continue
        visited.add(url)
 
        print(f"  [{i+1}/{len(level2_list)}] {url}")
 
        if is_pdf_url(url):
            text = download_and_extract_pdf(url)
            if text.strip():
                documents.append((url, text))
        else:
            # For level 2, we scrape text but do NOT follow further links
            text, _ = scrape_webpage(url)
            if text.strip():
                documents.append((url, text))
 
        total_pages += 1
        time.sleep(CRAWL_DELAY)
 
    print(f"\n  === CRAWL COMPLETE ===")
    print(f"  Total pages visited: {total_pages}")
    print(f"  Documents collected: {len(documents)}")
    print(f"  PDFs downloaded: {sum(1 for url, _ in documents if is_pdf_url(url))}")
 
    return documents
 
 
# ============================================================================
# STEP 5: TEXT PREPROCESSING
# ============================================================================
 
def remove_non_english(text):
    """
    Removes non-ASCII characters (Hindi/Devanagari, special symbols).
    Keeps English letters, digits, basic punctuation, and whitespace.
    """
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    return text
 
 
def preprocess_text(text):
    """
    Full preprocessing pipeline:
      1. Remove non-English characters
      2. Lowercase
      3. Remove URLs and email addresses
      4. Remove standalone numbers and excessive punctuation
      5. Tokenize using NLTK
      6. Filter short tokens (< 2 chars)
    Returns a list of cleaned tokens.
    """
    text = remove_non_english(text)
    text = text.lower()

    # Remove common web boilerplate words that add noise to the corpus
    boilerplate = "important links cccd iitj recruitment correspondence rti tenders techscape contact old website nccr portal how to reach iitj institute repository donations web policy web information manager feedback cert in help nirf internal committee intranet links copyright all rights reserved this portal is owned designed and developed by digital infrastructure automation iit jodhpur nagaur road karwar jodhpur rajasthan india for any comments enquiries feedback please email the wim"
    text = text.replace(boilerplate, " ")
    text = text.replace("_redirecttologinpage_", " ")
    text = re.sub(r'last updated \w+ [ap]m', ' ', text)
 
    # Remove URLs and emails
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
 
    # Remove standalone numbers and punctuation
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
 
    # Tokenize
    tokens = word_tokenize(text)
 
    # Keep only tokens with 2+ characters
    tokens = [t for t in tokens if len(t) >= 2]
 
    return tokens
 
 
def preprocess_table_text(text):
    """
    Special preprocessor for tabular data.
    Tables often have structured content like:
      "CS101 | Introduction to Programming | 4 Credits"
 
    We split on the pipe separator and process each cell,
    then recombine. This preserves the semantic relationships
    between cells in the same row.
    """
    rows = text.split("\n")
    all_tokens = []
 
    for row in rows:
        # Split by pipe separator (from our table extraction)
        cells = row.split("|")
        row_tokens = []
        for cell in cells:
            tokens = preprocess_text(cell)
            row_tokens.extend(tokens)
 
        if len(row_tokens) >= 2:
            all_tokens.append(row_tokens)
 
    return all_tokens
 
 
# ============================================================================
# STEP 6: BUILD CORPUS
# ============================================================================
 
def build_corpus(documents):
    """
    Processes all collected documents into a list of sentence-level
    token lists, which is the format Word2Vec expects.
 
    Strategy:
      - For regular text: split into sentences, tokenize each sentence
      - For PDF text: split into paragraphs, tokenize each paragraph
      - For tabular content: each row becomes a "sentence"
 
    Returns:
      - all_tokens: flat list for statistics
      - doc_token_lists: list of token lists for Word2Vec training
    """
    doc_token_lists = []
 
    for url, text in documents:
        # Split text into sentences for finer-grained training
        # Word2Vec works better with sentence-level input
        sentences = sent_tokenize(text)
 
        for sentence in sentences:
            tokens = preprocess_text(sentence)
            if len(tokens) >= 3:  # skip very short fragments
                doc_token_lists.append(tokens)
 
    # Flatten for statistics
    all_tokens = [t for doc in doc_token_lists for t in doc]
    return all_tokens, doc_token_lists
 
 
# ============================================================================
# STEP 7: STATISTICS AND WORD CLOUD
# ============================================================================
 
def compute_and_save_statistics(all_tokens, doc_token_lists, documents):
    """
    Computes and saves corpus statistics.
    Reports: sources, documents, tokens, vocabulary, top-30 words.
    """
    vocab = set(all_tokens)
    freq = Counter(all_tokens)
    top_30 = freq.most_common(30)
 
    # Count source types
    n_html = sum(1 for url, _ in documents if not is_pdf_url(url))
    n_pdf = sum(1 for url, _ in documents if is_pdf_url(url))
 
    stats_text = (
        "=" * 60 + "\n"
        "CORPUS STATISTICS\n"
        "=" * 60 + "\n"
        f"Source pages scraped (HTML):        {n_html}\n"
        f"Source PDFs downloaded:             {n_pdf}\n"
        f"Total source documents:             {len(documents)}\n"
        f"Total sentences (for Word2Vec):     {len(doc_token_lists)}\n"
        f"Total tokens:                       {len(all_tokens)}\n"
        f"Vocabulary size (unique tokens):    {len(vocab)}\n"
        f"Avg tokens per sentence:            {len(all_tokens) / max(len(doc_token_lists), 1):.1f}\n"
        "\nTop 30 most frequent words:\n"
        "-" * 40 + "\n"
    )
    for rank, (word, count) in enumerate(top_30, 1):
        stats_text += f"  {rank:3d}. {word:<25s} {count}\n"
 
    stats_path = os.path.join(OUTPUT_DIR, "corpus_stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
 
    print(stats_text)
    return freq
 
 
def generate_wordcloud(freq_dict):
    """
    Creates a word cloud from word frequencies, excluding stopwords.
    Saves as a high-res PNG for the report.
    """
    stop_words = set(stopwords.words('english'))
    # Also remove some common web boilerplate words
    extra_stops = {"iit", "jodhpur", "page", "click", "home", "menu",
                   "search", "login", "website", "http", "https", "www",
                   "copyright", "reserved", "rights", "email", "phone",
                   "address", "contact"}
    stop_words.update(extra_stops)
 
    filtered_freq = {w: c for w, c in freq_dict.items()
                     if w not in stop_words and len(w) > 2}
 
    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        max_words=150,
        colormap="viridis",
        contour_width=1,
        contour_color="steelblue"
    )
    wc.generate_from_frequencies(filtered_freq)
 
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud — IIT Jodhpur Corpus", fontsize=16)
 
    wc_path = os.path.join(OUTPUT_DIR, "wordcloud.png")
    plt.savefig(wc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Word cloud saved to {wc_path}\n")
 
 
# ============================================================================
# STEP 8: SAVE CORPUS
# ============================================================================
 
def save_corpus(doc_token_lists):
    """
    Saves cleaned corpus as a text file.
    Each line = one sentence (space-separated tokens).
    This is directly consumable by gensim's Word2Vec.
    """
    corpus_path = os.path.join(OUTPUT_DIR, "cleaned_corpus.txt")
    with open(corpus_path, "w") as f:
        for tokens in doc_token_lists:
            f.write(" ".join(tokens) + "\n")
    print(f"  Corpus saved to {corpus_path}")
    print(f"  Total lines: {len(doc_token_lists)}\n")
 
 
def save_source_log(documents):
    """
    Saves a log of all sources crawled — useful for the report
    to show which sources your data came from.
    """
    log_path = os.path.join(OUTPUT_DIR, "source_log.txt")
    with open(log_path, "w") as f:
        f.write("CRAWLED SOURCES LOG\n")
        f.write("=" * 60 + "\n\n")
        for i, (url, text) in enumerate(documents, 1):
            doc_type = "PDF" if is_pdf_url(url) else "HTML"
            token_count = len(preprocess_text(text))
            f.write(f"{i:4d}. [{doc_type}] {url}\n")
            f.write(f"       Tokens extracted: ~{token_count}\n\n")
    print(f"  Source log saved to {log_path}")
 
 
# ============================================================================
# MAIN EXECUTION
# ============================================================================
 
if __name__ == "__main__":
    print("=" * 60)
    print("TASK 1: DATA COLLECTION (v2 — Auto-Crawling)")
    print("=" * 60)
 
    # 1. Crawl
    print("\n[1/5] Starting 2-level crawl of IIT Jodhpur websites...")
    documents = crawl(URLS)
 
    if not documents:
        print("[ERROR] No documents collected! Check your internet connection.")
        exit(1)
 
    # 2. Build corpus
    print("\n[2/5] Preprocessing and building corpus...")
    all_tokens, doc_token_lists = build_corpus(documents)
 
    if not all_tokens:
        print("[ERROR] Preprocessing produced no tokens!")
        exit(1)
 
    # 3. Statistics
    print("\n[3/5] Computing corpus statistics...")
    freq = compute_and_save_statistics(all_tokens, doc_token_lists, documents)
 
    # 4. Word cloud
    print("\n[4/5] Generating word cloud...")
    generate_wordcloud(freq)
 
    # 5. Save everything
    print("\n[5/5] Saving corpus and logs...")
    save_corpus(doc_token_lists)
    save_source_log(documents)
 
    print("\n" + "=" * 60)
    print("Task 1 complete! Files saved in output/")
    print("  - cleaned_corpus.txt  (for Word2Vec training)")
    print("  - corpus_stats.txt    (for report)")
    print("  - wordcloud.png       (for report)")
    print("  - source_log.txt      (list of all crawled sources)")
    print("=" * 60)