# Tyre-y Devs — Guaraní-Spanish Language Dataset & Model Suite

Este repositorio contiene la arquitectura completa para la recolección, generación sintética, limpieza, validación y fine-tuning de modelos para la traducción y procesamiento del idioma **Guaraní** en relación con el **Español**.

---

## 📁 Estructura del Repositorio

```text
.
├── data/
│   ├── raw/           # Corpus original extraído y en bruto (CSV y JSONL)
│   ├── synthetic/      # Datos generados por LLMs (sin revisar)
│   └── validated/      # Datos aprobados por el equipo lingüístico (Linguist Hero)
├── scraper/            # Módulo de Web Scraping automático
│   ├── main_scraper.py # Script ejecutable de extracción
│   └── requirements.txt
├── prompts/            # Plantillas de prompts para generación sintética (Módulo 3)
├── src/                # Scripts de procesamiento, generación y limpieza (Módulo 4)
├── notebooks/          # Notebooks de Google Colab para fine-tuning con QLoRA (Módulo 2)
├── reports/            # Informe técnico en PDF (Módulo 5)
├── slides/             # Presentación del Demo Day
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Módulo de Web Scraping (`/scraper/`)

El scraper extrae corpus bilingüe Guaraní - Español desde fuentes web, aplicando filtros de calidad en tiempo real y guardando los datos por lotes (batching) con deduplicación automática.

### Requisitos e Instalación

```bash
pip install -r requirements.txt
```

### Ejecución del Scraper

1. **Ejecutar con fuentes por defecto:**
   ```bash
   python scraper/main_scraper.py
   ```

2. **Scrapear una URL específica dada por el equipo:**
   ```bash
   python scraper/main_scraper.py --url "https://ejemplo.com/glosario" --categoria "diccionario"
   ```

3. **Cargar una lista de URLs desde un JSON:**
   ```bash
   python scraper/main_scraper.py --urls-file mis_urls.json
   ```

### Archivos de Salida (`data/raw/`)
- `scraped_dataset.csv`: Dataset en formato tabular visible.
- `scraped_dataset.jsonl`: Formato estándar de líneas JSON listo para fine-tuning e integración con HuggingFace / Colab.
