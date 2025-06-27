FROM python:3.10-slim

ADD . .

RUN pip install -r requirements.txt

WORKDIR /src

EXPOSE 6666

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6666"]