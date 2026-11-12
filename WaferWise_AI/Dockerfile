# WaferWise AI - Streamlit app image.
# Python 3.12 is required by the current numpy / scipy / xgboost / shap releases.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# libgomp is required by XGBoost / scikit-learn OpenMP code
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (uid 1000, also what Hugging Face Spaces expects)
RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY --chown=appuser:appuser requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=appuser:appuser . .
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
