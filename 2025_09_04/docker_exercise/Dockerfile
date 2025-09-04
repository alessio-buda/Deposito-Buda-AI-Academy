
# Usa un'immagine ufficiale di Python come base
FROM python:3.11

# Crea e imposta la directory di lavoro nel container
WORKDIR /app

# Copia il certificato SSL nel container
COPY cacert.pem /usr/local/share/ca-certificates/cacert.crt

# Aggiorna i certificati del sistema
RUN apt-get update && apt-get install -y ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Imposta le variabili d'ambiente per SSL
ENV REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/cacert.crt
ENV SSL_CERT_FILE=/usr/local/share/ca-certificates/cacert.crt
ENV CURL_CA_BUNDLE=/usr/local/share/ca-certificates/cacert.crt

# Copia i file requirements.txt
COPY requirements.txt .

# Installa le dipendenze usando il certificato custom
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --no-cache-dir -r requirements.txt

# Copia il file dell'app nel container
COPY app.py .

# Comando che Docker esegue quando il container parte
CMD ["python", "app.py"]
