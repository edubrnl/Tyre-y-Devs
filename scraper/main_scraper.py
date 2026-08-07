#!/usr/bin/env python3
"""
Main Web Scraper for Guaraní - Spanish Corpus Extraction
Module: /scraper/main_scraper.py

Features:
- Requests with custom user-agents & exponential backoff retries.
- HTML cleaning (scripts, styles, special chars, preserving Guaraní diacritics).
- Extraction from <table>, <ul>/<ol>/<li>, <dl>/<dt>/<dd>, and structured text pairs.
- In-situ quality filters: length check (< 10 chars), deduplication across runs.
- Batch saving every N items to ../data/raw/scraped_dataset.csv and scraped_dataset.jsonl.
- Progress bar and visible logging.
"""

import os
import re
import json
import uuid
import hashlib
import time
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
from tqdm import tqdm
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GuaraniScraper")

# ---------------------------------------------------------------------------
# Constants & Settings
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = (BASE_DIR.parent / "data" / "raw").resolve()

OUTPUT_CSV = DATA_RAW_DIR / "scraped_dataset.csv"
OUTPUT_JSONL = DATA_RAW_DIR / "scraped_dataset.jsonl"

BATCH_SIZE = 50
MIN_CHAR_LENGTH = 10

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0"
]

# Delimiters commonly used to separate Spanish and Guaraní translation pairs in list items / text lines
DELIMITERS_PATTERN = re.compile(r"\s*(?:\s–\s|\s—\s|\s-\s|\s:\s|\s=\s|\s->\s|\s/|\t|:|-|=)\s*")

# ---------------------------------------------------------------------------
# HTTP Session Factory
# ---------------------------------------------------------------------------
def create_retry_session(
    retries: int = 5,
    backoff_factor: float = 1.5,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504)
) -> requests.Session:
    """
    Creates a requests.Session configured with automatic retries and exponential backoff.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Default Headers
    session.headers.update({
        "User-Agent": USER_AGENTS[0],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9,gn;q=0.8,en;q=0.7",
        "Connection": "keep-alive"
    })
    return session

# ---------------------------------------------------------------------------
# Text Cleaning & Normalization Helpers
# ---------------------------------------------------------------------------
def clean_html_soup(soup: BeautifulSoup) -> None:
    """Removes scripts, styles, metadata, and comments in-place from BeautifulSoup object."""
    for element in soup(["script", "style", "noscript", "header", "footer", "nav", "iframe"]):
        element.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

def clean_text(text: Optional[str]) -> str:
    """
    Cleans raw extracted string:
    - Normalizes whitespace (multiple spaces/tabs/newlines into a single space).
    - Preserves standard Spanish and Guaraní characters (accents, tildes, puso/apostrophes).
    """
    if not text:
        return ""
    # Unify line breaks and tabs to spaces
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Remove HTML entities leftover or weird non-printable control characters
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def compute_pair_hash(es_text: str, gn_text: str) -> str:
    """Generates a deterministic unique hash for a sentence pair."""
    normalized = f"{es_text.strip().lower()}|||{gn_text.strip().lower()}"
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()

NOISE_PATTERNS = re.compile(
    r"(?:^↑|control de autoridades|obtenido de|issn\s*\d|doi:|isbn|consultado el|artículo principal|proyectos wikimedia|identificadores gnd|multimedia:)",
    re.IGNORECASE
)

def is_valid_pair(es_text: str, gn_text: str, min_len: int = MIN_CHAR_LENGTH) -> bool:
    """
    In-situ quality filter:
    - Omit sentences/phrases shorter than `min_len` characters in either language.
    - Check that strings contain actual words and are not purely numerical or punctuation.
    - Omit Wikipedia metadata, footnotes, citations, and URLs.
    """
    if len(es_text) < min_len or len(gn_text) < min_len:
        return False
    # Check if text contains at least some alphanumeric characters
    if not re.search(r"\w", es_text) or not re.search(r"\w", gn_text):
        return False
    # Filter out citation noise and metadata
    if NOISE_PATTERNS.search(es_text) or NOISE_PATTERNS.search(gn_text):
        return False
    if "http://" in es_text.lower() or "https://" in es_text.lower() or "http://" in gn_text.lower() or "https://" in gn_text.lower():
        return False
    return True

# ---------------------------------------------------------------------------
# Scraping Parsers
# ---------------------------------------------------------------------------
def parse_tables(soup: BeautifulSoup, url: str, category: str) -> List[Dict[str, Any]]:
    """Extracts parallel sentence/glossary pairs from <table> structures."""
    results = []
    tables = soup.find_all("table")
    
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cols = [clean_text(col.get_text()) for col in row.find_all(["td", "th"])]
            if len(cols) >= 2:
                col_a, col_b = cols[0], cols[1]
                # Filter out headers like "Español", "Guaraní"
                if col_a.lower() in ["español", "espanol", "spanish", "castellano"] or \
                   col_b.lower() in ["guaraní", "guarani"]:
                    continue
                
                # Check heuristic order or treat col_a as Spanish, col_b as Guaraní
                if is_valid_pair(col_a, col_b):
                    results.append({
                        "texto_espanol": col_a,
                        "texto_guarani": col_b,
                        "fuente_url": url,
                        "categoria": category
                    })
    return results

def parse_lists(soup: BeautifulSoup, url: str, category: str) -> List[Dict[str, Any]]:
    """Extracts translation pairs from <ul>, <ol>, <li> tags using split delimiters."""
    results = []
    items = soup.find_all("li")
    
    for item in items:
        raw_text = clean_text(item.get_text())
        parts = DELIMITERS_PATTERN.split(raw_text, maxsplit=1)
        if len(parts) == 2:
            es_cand, gn_cand = clean_text(parts[0]), clean_text(parts[1])
            if is_valid_pair(es_cand, gn_cand):
                results.append({
                    "texto_espanol": es_cand,
                    "texto_guarani": gn_cand,
                    "fuente_url": url,
                    "categoria": category
                })
    return results

def parse_definition_lists(soup: BeautifulSoup, url: str, category: str) -> List[Dict[str, Any]]:
    """Extracts pairs from <dl> (Definition List), <dt> (Term), <dd> (Definition) tags."""
    results = []
    dls = soup.find_all("dl")
    
    for dl in dls:
        dts = dl.find_all("dt")
        for dt in dts:
            dt_text = clean_text(dt.get_text())
            # Find next sibling dd
            dd = dt.find_next_sibling("dd")
            if dd:
                dd_text = clean_text(dd.get_text())
                if is_valid_pair(dt_text, dd_text):
                    results.append({
                        "texto_espanol": dt_text,
                        "texto_guarani": dd_text,
                        "fuente_url": url,
                        "categoria": category
                    })
    return results

def parse_paragraphs_or_divs(soup: BeautifulSoup, url: str, category: str) -> List[Dict[str, Any]]:
    """Fallback parser for paragraph or div lines separated by delimiters."""
    results = []
    elements = soup.find_all(["p", "div", "blockquote"])
    
    for el in elements:
        # Avoid processing large blocks containing child elements like lists/tables
        if el.find(["table", "ul", "ol", "dl"]):
            continue
        raw_text = clean_text(el.get_text())
        parts = DELIMITERS_PATTERN.split(raw_text, maxsplit=1)
        if len(parts) == 2:
            es_cand, gn_cand = clean_text(parts[0]), clean_text(parts[1])
            if is_valid_pair(es_cand, gn_cand):
                results.append({
                    "texto_espanol": es_cand,
                    "texto_guarani": gn_cand,
                    "fuente_url": url,
                    "categoria": category
                })
    return results

# ---------------------------------------------------------------------------
# Storage & Batch Persistence Manager
# ---------------------------------------------------------------------------
class DatasetStorageManager:
    """Handles thread-safe/batch saving to CSV and JSONL files with deduplication."""
    def __init__(self, csv_path: Path, jsonl_path: Path):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.seen_hashes: Set[str] = set()
        self._ensure_output_dir()
        self._load_existing_hashes()

    def _ensure_output_dir(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_existing_hashes(self):
        """Loads existing hashes from JSONL or CSV if files already exist to prevent re-scraping duplicates."""
        if self.jsonl_path.exists():
            try:
                with open(self.jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            es = record.get("texto_espanol", "")
                            gn = record.get("texto_guarani", "")
                            if es and gn:
                                self.seen_hashes.add(compute_pair_hash(es, gn))
                logger.info(f"Loaded {len(self.seen_hashes)} existing unique records from {self.jsonl_path.name}")
            except Exception as e:
                logger.warning(f"Error reading existing JSONL file: {e}")

    def is_duplicate(self, es_text: str, gn_text: str) -> bool:
        pair_hash = compute_pair_hash(es_text, gn_text)
        if pair_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(pair_hash)
        return False

    def save_batch(self, batch_data: List[Dict[str, Any]]) -> int:
        """
        Appends a batch of items to both CSV and JSONL formats.
        Returns the count of new saved items.
        """
        if not batch_data:
            return 0

        now_iso = datetime.now(timezone.utc).isoformat()
        records_to_save = []

        for item in batch_data:
            es = item["texto_espanol"]
            gn = item["texto_guarani"]

            if self.is_duplicate(es, gn):
                continue

            record = {
                "id": str(uuid.uuid4()),
                "fuente_url": item["fuente_url"],
                "texto_espanol": es,
                "texto_guarani": gn,
                "categoria": item.get("categoria", "general"),
                "fecha_crawleo": now_iso
            }
            records_to_save.append(record)

        if not records_to_save:
            return 0

        # Save to JSONL (append line by line)
        with open(self.jsonl_path, "a", encoding="utf-8") as f_jsonl:
            for rec in records_to_save:
                f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Save to CSV (append using Pandas)
        df_new = pd.DataFrame(records_to_save)
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        df_new.to_csv(
            self.csv_path,
            mode="a" if file_exists else "w",
            header=not file_exists,
            index=False,
            encoding="utf-8"
        )

        return len(records_to_save)

# ---------------------------------------------------------------------------
# Default Target URLs Configuration
# ---------------------------------------------------------------------------
# Target websites containing parallel Guarani-Spanish texts, dictionaries, and phrasebooks.
DEFAULT_TARGETS = [
    {
        "url": "https://es.wikipedia.org/wiki/Idioma_guaran%C3%AD",
        "categoria": "gramatica_y_vocabulario"
    },
    {
        "url": "https://es.wikipedia.org/wiki/Alfabeto_guaran%C3%AD",
        "categoria": "alfabeto"
    },
    {
        "url": "https://es.wiktionary.org/wiki/Categor%C3%ADa:Guaran%C3%AD",
        "categoria": "diccionario"
    },
    {
        "url": "https://es.wikibooks.org/wiki/Guaran%C3%AD",
        "categoria": "curso"
    }
]

# ---------------------------------------------------------------------------
# Core Scraper Controller
# ---------------------------------------------------------------------------
class GuaraniWebScraper:
    def __init__(self, targets: Optional[List[Dict[str, str]]] = None):
        self.targets = targets or DEFAULT_TARGETS
        self.session = create_retry_session()
        self.storage = DatasetStorageManager(OUTPUT_CSV, OUTPUT_JSONL)

    def fetch_url(self, url: str) -> Optional[str]:
        """Fetches HTML content with error handling and randomized user agents."""
        try:
            # Rotate user agent
            headers = {"User-Agent": USER_AGENTS[hash(url) % len(USER_AGENTS)]}
            response = self.session.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                response.encoding = response.apparent_encoding or "utf-8"
                return response.text
            else:
                logger.warning(f"Failed to fetch {url} - Status Code: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error requesting {url}: {e}")
            return None

    def preview_url(self, url: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Preview mode: Fetches a URL and prints a human-readable table of sample pairs
        found on the webpage BEFORE saving.
        """
        logger.info(f"\n--- PREVISUALIZANDO URL: {url} ---")
        html_content = self.fetch_url(url)
        if not html_content:
            logger.error("No se pudo obtener el contenido de la URL.")
            return []

        soup = BeautifulSoup(html_content, "html.parser")
        clean_html_soup(soup)

        pairs = []
        pairs.extend(parse_tables(soup, url, "preview"))
        pairs.extend(parse_lists(soup, url, "preview"))
        pairs.extend(parse_definition_lists(soup, url, "preview"))
        pairs.extend(parse_paragraphs_or_divs(soup, url, "preview"))

        print("\n" + "="*80)
        print(f" RESULTADOS ENCONTRADOS EN LA PÁGINA ({len(pairs)} pares detectados)")
        print("="*80)

        if not pairs:
            print("⚠️ No se detectaron pares Guaraní-Español con la estructura estándar en esta página.")
            print("Pistas: Verifica si la página usa un formato de tabla <table> o lista <li>.")
        else:
            print(f"Muestra de los primeros {min(limit, len(pairs))} pares extraídos:\n")
            for idx, p in enumerate(pairs[:limit], 1):
                print(f"  [{idx}] ESPAÑOL: {p['texto_espanol']}")
                print(f"      GUARANÍ: {p['texto_guarani']}")
                print("-" * 80)

        print("="*80 + "\n")
        return pairs

    def run(self):
        """Executes the web scraping workflow across target URLs."""
        logger.info("Starting Guaraní - Spanish Web Scraper...")
        logger.info(f"Target URLs count: {len(self.targets)}")
        logger.info(f"Output CSV: {OUTPUT_CSV}")
        logger.info(f"Output JSONL: {OUTPUT_JSONL}")

        total_extracted = 0
        total_saved = 0
        buffer: List[Dict[str, Any]] = []

        progress_bar = tqdm(self.targets, desc="Scraping Target URLs", unit="site")

        for target in progress_bar:
            url = target["url"]
            categoria = target.get("categoria", "general")
            progress_bar.set_postfix({"current_url": url[:30] + "..."})

            html_content = self.fetch_url(url)
            if not html_content:
                continue

            soup = BeautifulSoup(html_content, "html.parser")
            clean_html_soup(soup)

            extracted_pairs = []
            # Apply all modular parsers
            extracted_pairs.extend(parse_tables(soup, url, categoria))
            extracted_pairs.extend(parse_lists(soup, url, categoria))
            extracted_pairs.extend(parse_definition_lists(soup, url, categoria))
            extracted_pairs.extend(parse_paragraphs_or_divs(soup, url, categoria))

            total_extracted += len(extracted_pairs)
            buffer.extend(extracted_pairs)

            # Checkpoint batch save when buffer reaches BATCH_SIZE
            while len(buffer) >= BATCH_SIZE:
                batch_to_save = buffer[:BATCH_SIZE]
                buffer = buffer[BATCH_SIZE:]
                saved_count = self.storage.save_batch(batch_to_save)
                total_saved += saved_count

            # Polite crawl delay between domains
            time.sleep(1.0)

        # Save remaining buffer items
        if buffer:
            saved_count = self.storage.save_batch(buffer)
            total_saved += saved_count

        logger.info("==================================================")
        logger.info(f"Scraping Completed successfully!")
        logger.info(f"Total raw candidate pairs extracted: {total_extracted}")
        logger.info(f"Total unique valid pairs saved: {total_saved}")
        logger.info(f"Dataset location: {DATA_RAW_DIR}")
        logger.info("==================================================")

# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web Scraper para extracción de corpus bilingüe Guaraní - Español")
    parser.add_argument("--url", type=str, help="URL específica para scrapear")
    parser.add_argument("--categoria", type=str, default="custom", help="Categoría para la URL específica")
    parser.add_argument("--urls-file", type=str, help="Archivo JSON con lista de URLs en formato [{'url': ..., 'categoria': ...}]")
    parser.add_argument("--preview", action="store_true", help="Modo previsualización: Muestra una muestra de lo encontrado sin guardar")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo: Te guía paso a paso pegando una URL")

    args = parser.parse_args()
    scraper = GuaraniWebScraper()

    if args.interactive or (not args.url and not args.urls_file and not args.preview and len(os.sys.argv) == 1):
        print("\n=== MODO INTERACTIVO ASISTIDO DEL SCRAPER ===")
        print("Pega la URL de la pagina web de la que queres extraer informacion Guarani-Espanol.")
        try:
            target_url = input("\n- Ingresa la URL (o presiona ENTER para usar las fuentes predeterminadas): ").strip()
        except EOFError:
            target_url = ""
        
        if target_url:
            try:
                cat = input("- Ingresa la categoria (ej: diccionario, frases, gramatica) [default: general]: ").strip() or "general"
            except EOFError:
                cat = "general"
            sample_pairs = scraper.preview_url(target_url)
            
            if sample_pairs:
                try:
                    confirm = input("¿Deseas guardar los datos extraidos de esta pagina en el dataset? (S/n): ").strip().lower()
                except EOFError:
                    confirm = "s"
                if confirm in ["", "s", "si", "sí", "y", "yes"]:
                    saved = scraper.storage.save_batch(sample_pairs)
                    print(f"OK: Guardados {saved} pares unicos en data/raw!")
                else:
                    print("Cancelado. No se guardaron los datos.")
        else:
            print("\nProcesando fuentes predeterminadas...")
            scraper.run()

    elif args.preview and args.url:
        scraper.preview_url(args.url)

    else:
        custom_targets = None
        if args.url:
            custom_targets = [{"url": args.url, "categoria": args.categoria}]
        elif args.urls_file:
            file_path = Path(args.urls_file)
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    custom_targets = json.load(f)
            else:
                logger.error(f"El archivo {args.urls_file} no existe.")

        scraper = GuaraniWebScraper(targets=custom_targets)
        scraper.run()
