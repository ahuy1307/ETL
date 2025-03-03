FROM python:3.9.16-slim-buster

# Sử dụng các biến môi trường từ bên ngoài
ENV BASE_URL=${BASE_URL}
ENV PAGE_SIZE=${PAGE_SIZE}
ENV PROJECT_ID=${PROJECT_ID}
ENV DATASET_ID=${DATASET_ID}
ENV TABLE_ID=${TABLE_ID}

# Cấu hình Service Account từ biến môi trường
ENV TYPE=${TYPE}
ENV PRIVATE_KEY_ID=${PRIVATE_KEY_ID}
ENV PRIVATE_KEY=${PRIVATE_KEY}
ENV CLIENT_EMAIL=${CLIENT_EMAIL}
ENV CLIENT_ID=${CLIENT_ID}
ENV AUTH_URI=${AUTH_URI}
ENV TOKEN_URI=${TOKEN_URI}
ENV AUTH_PROVIDER_X509_CERT_URL=${AUTH_PROVIDER_X509_CERT_URL}
ENV CLIENT_X509_CERT_URL=${CLIENT_X509_CERT_URL}
ENV UNIVERSE_DOMAIN=${UNIVERSE_DOMAIN}


COPY . /api

RUN pip install -r /api/requirements.txt

WORKDIR /api

COPY service-account.json /app/service-account.json

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

