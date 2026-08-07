#!/usr/bin/env python3
"""
Downloader for Guaran-IA Corpus (GitHub Git-LFS Dataset)
Module: /src/download_guaran_ia_corpus.py
"""

import sys
import logging
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GuaranIACorpusDownloader")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"

# Base Git LFS media URL for guaran-ia/corpus
LFS_BASE_URL = "https://media.githubusercontent.com/media/guaran-ia/corpus/main/data/raw"

KEY_DATASETS = [
    "gua_spa.jsonl",
    "Alpaca-gn-gpt4.jsonl",
    "grammar.jsonl",
    "tatoeba.jsonl",
    "flores-200.jsonl",
    "guarania-scraped.jsonl"
]

def download_file(filename: str) -> bool:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_RAW_DIR / filename
    url = f"{LFS_BASE_URL}/{filename}"

    logger.info(f"Descargando {filename} desde Guaran-IA (Git LFS)...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(out_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"✅ ¡Guardado con éxito en: {out_path}")
            return True
        else:
            logger.error(f"❌ Error HTTP {response.status_code} al descargar {url}")
            return False
    except Exception as e:
        logger.error(f"❌ Excepción al descargar {filename}: {e}")
        return False

def download_all_key_datasets():
    logger.info("Iniciando descarga del repositorio Guaran-IA Corpus...")
    successful = 0
    for fn in KEY_DATASETS:
        if download_file(fn):
            successful += 1
    logger.info(f"Completado: {successful}/{len(KEY_DATASETS)} archivos descargados en data/raw/")

if __name__ == "__main__":
    download_all_key_datasets()
