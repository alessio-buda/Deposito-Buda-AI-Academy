
FROM python:3.11

WORKDIR /app

# Copy the certificate bundle from your local system
COPY cacert.pem /usr/local/share/ca-certificates/cacert.crt
RUN update-ca-certificates

# Set environment variables to use the certificates
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

COPY requirements_cross_platform.txt requirements.txt

# Install packages with additional SSL configuration
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host github.com --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["python", "app.py"]
