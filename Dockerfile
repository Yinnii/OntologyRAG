FROM python:3.10-slim

ADD . .

RUN pip install -r requirements.txt

WORKDIR /src

ENV NEO4J_URI="neo4j://localhost:7688"
ENV NEO4J_USER="neo4j"
ENV NEO4J_PASSWORD="NEO4J_PASSWORD"
ENV POSTGRES_PASSWORD="POSTGRES_PASSWORD"
ENV POSTGRES_USER="postgres"
ENV POSTGRES_DB="openml"
ENV POSTGRES_HOST="localhost"
ENV POSTGRES_PORT="5434"
ENV EMBEDDING_LARGE="https://...openai.azure.com/"
ENV EMBEDDING_SMALL="https://...openai.azure.com/"
ENV AZURE_ENDPOINT="https://...openai.azure.com/"
ENV AZURE_API_KEY='AZURE_API_KEY'
ENV OPENAI_API_KEY='OPENAI_API_KEY'

EXPOSE 6666

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6666"]