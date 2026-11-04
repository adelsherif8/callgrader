# CPU image for the grading API. Mount or copy a trained model into /app/models/encoder.
FROM python:3.12-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 \
 && pip install -r requirements.txt
COPY callgrader callgrader
COPY serve serve
COPY data/test.jsonl data/test.jsonl
RUN useradd -m grader && chown -R grader /app
USER grader
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
