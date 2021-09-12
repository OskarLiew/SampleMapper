FROM python:3.8

WORKDIR /app

COPY ./sample_mapper ./sample_mapper
RUN pip install -r ./sample_mapper/requirements.txt

EXPOSE 8050

CMD ["python", "./sample_mapper/app.py", "&"]