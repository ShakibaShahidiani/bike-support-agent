# One image, two services (see compose.yaml): the API (uvicorn) and the UI (streamlit).
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent agent
COPY api api
COPY ui ui
COPY data data
COPY prompts prompts
COPY .streamlit .streamlit
# Vector index built locally (python -m agent.rag) and shipped in the image.
COPY chroma_db chroma_db
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
