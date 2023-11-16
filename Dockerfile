# Verwenden Sie ein offizielles Python-Laufzeitbild als Elternbild
FROM python:3.10-slim-buster

# Setzen Sie das Arbeitsverzeichnis im Container auf /app
WORKDIR /app

# Kopieren Sie die aktuellen Verzeichnisinhalte in das Arbeitsverzeichnis im Container
COPY . /app

# Installieren Sie alle benötigten Pakete
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Führen Sie die Anwendung aus, wenn der Container gestartet wird
CMD ["python", "app.py"]