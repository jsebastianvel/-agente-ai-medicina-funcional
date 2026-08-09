FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1 \
    CHROMA_DATA_DIR=/app/data/chroma \
    LOG_DIR=/app/logs

# Bake the RAG index into the image so the container is immediately answer-ready.
# GEMINI_API_KEY must be available at build time for the embedding calls.
ARG GEMINI_API_KEY
ENV GEMINI_API_KEY=${GEMINI_API_KEY}
RUN python -m agente_ai.ingestion.build_index

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
