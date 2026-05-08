# RAG local (sin LLM generativo)

Este proyecto implementa un **RAG local** sobre la base documental en `docs/Wiki`: indexa contenido, recupera contexto relevante y responde de forma **extractiva** con citas, sin usar un LLM generativo externo.
La “AI local” del sistema se basa en **modelos de embeddings y reranking ejecutados en local** (Hugging Face/Sentence-Transformers), combinados con búsqueda léxica en PostgreSQL para mejorar precisión.

RAG local sobre documentos `docs/Wiki` con:
- **FastAPI** (API)
- **PostgreSQL + pgvector** (persistencia + búsqueda vectorial)
- **NLP local**: búsqueda léxica con `tsvector/plainto_tsquery` (español)
- **HuggingFace local**: reranking con `sentence-transformers` (cross-encoder)
- **Streamlit** (frontend)
- **Memoria por conversación** en tablas `conversations/messages`

La respuesta no usa un LLM generativo: se arma de forma extractiva con los chunks recuperados y sus citas.

## Estructura relevante

- `main.py`: bootstrap de app, DB y rutas.
- `src/services/rag.py`: ingesta `.mhtml`, embeddings locales, retrieval híbrido y respuesta extractiva.
- `src/services/chat.py`: memoria conversacional.
- `src/api/routes.py`: endpoints `/health`, `/ingest`, `/chat`, `/history/{conversation_id}`.
- `streamlit_app.py`: UI local.

## Ejecutar con contenedores

1. Levantar stack:
   ```bash
   docker compose up --build
   ```
2. API en `http://localhost:8000`, UI en `http://localhost:8501`.
3. Indexar documentos:
   ```bash
   curl -X POST http://localhost:8000/ingest
   ```
   Reindexado forzado:
   ```bash
   curl -X POST "http://localhost:8000/ingest?rebuild=true"
   ```
   También puedes usar el botón **Indexar documentos** en Streamlit.
4. Chatear desde Streamlit o por API:
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"content":"¿Cómo configurar Asterisk?"}'
   ```

## Variables de entorno

Config base en `.env`:
- `DATABASE_URL`
- `DOCS_PATH`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIM`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVAL_TOP_K`
- `RETRIEVAL_CANDIDATE_POOL`
- `HISTORY_WINDOW`
- `ENABLE_RERANKER`
- `RERANKER_MODEL`
- `CORS_ORIGINS`

## Nota de operación

- Si la base documental está vacía, el endpoint `/chat` dispara una indexación automática inicial.
- Tras cambios en limpieza/retrieval, reindexa con `POST /ingest` para regenerar chunks.
- `indexed_chunks: 0` no siempre es error: puede significar que todos los archivos fueron omitidos por no tener cambios.
  Revisa también:
  - `scanned_files` (archivos detectados)
  - `skipped_unchanged_files`
  - `failed_files`
  - `failures` (lista de archivos con error)
  - `total_chunks_in_db`
  - `docs_path_exists`
