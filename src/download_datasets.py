#!/usr/bin/env python3
"""
Dataset Downloader for Hugging Face Guaraní / Jopara Corpora
Module: /src/download_datasets.py
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetDownloader")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"

FILES_TO_DOWNLOAD = [
    {
        "url": "https://huggingface.co/datasets/mmaguero/gn-jopara-sentiment-analysis/raw/main/balanced/sa3_trainBal.txt",
        "name": "sa3_trainBal.txt",
        "categoria": "jopara_sentiment_balanced"
    },
    {
        "url": "https://huggingface.co/datasets/mmaguero/gn-jopara-sentiment-analysis/raw/main/balanced/sa3_devBal.txt",
        "name": "sa3_devBal.txt",
        "categoria": "jopara_sentiment_balanced"
    },
    {
        "url": "https://huggingface.co/datasets/mmaguero/gn-jopara-sentiment-analysis/raw/main/balanced/sa3_testBal.txt",
        "name": "sa3_testBal.txt",
        "categoria": "jopara_sentiment_balanced"
    },
    {
        "url": "https://huggingface.co/datasets/mmaguero/gn-jopara-sentiment-analysis/raw/main/unbalanced/sa3_train.txt",
        "name": "sa3_train.txt",
        "categoria": "jopara_sentiment_unbalanced"
    }
]

def download_and_process():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = DATA_RAW_DIR / "jopara_dataset.csv"
    out_jsonl = DATA_RAW_DIR / "jopara_dataset.jsonl"

    all_records = []
    now_iso = datetime.now(timezone.utc).isoformat()

    logger.info("Iniciando descarga de datasets desde Hugging Face...")

    for file_info in FILES_TO_DOWNLOAD:
        url = file_info["url"]
        logger.info(f"Descargando {file_info['name']}...")
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                lines = r.text.splitlines()
                count = 0
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Split line if formatted as label\ttext or sentence
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        label, text = parts[0].strip(), parts[1].strip()
                    else:
                        label, text = "neutral", line

                    record = {
                        "id": str(uuid.uuid4()),
                        "fuente_url": url,
                        "texto_guarani": text,
                        "etiqueta_sentimiento": label,
                        "categoria": file_info["categoria"],
                        "fecha_descarga": now_iso
                    }
                    all_records.append(record)
                    count += 1
                logger.info(f"Procesadas {count} oraciones de {file_info['name']}")
            else:
                logger.warning(f"Error {r.status_code} al descargar {url}")
        except Exception as e:
            logger.error(f"Excepción descargando {url}: {e}")

    if all_records:
        # Save JSONL
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Save CSV
        df = pd.DataFrame(all_records)
        df.to_csv(out_csv, index=False, encoding="utf-8")

        logger.info("==================================================")
        logger.info(f"¡Descarga y procesamiento completados!")
        logger.info(f"Total oraciones en Guaraní/Jopara guardadas: {len(all_records)}")
        logger.info(f"Archivos guardados en:")
        logger.info(f" - CSV:   {out_csv}")
        logger.info(f" - JSONL: {out_jsonl}")
        logger.info("==================================================")

if __name__ == "__main__":
    download_and_process()
