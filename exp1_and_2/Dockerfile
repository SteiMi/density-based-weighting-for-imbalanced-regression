FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN mkdir /app

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY *.py /app/

RUN chmod -R 777 /app

CMD [ "python", "run.py" ]
