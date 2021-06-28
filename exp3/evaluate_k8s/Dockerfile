FROM INSERT_YOUR_REGISTRY/tf-gdal-nogpu:latest


RUN mkdir -p /app/configs
RUN mkdir -p /app/out
COPY evaluate_k8s/evaluate.py /app/
COPY *.py /app/
COPY data /app/data
COPY configs/* /app/configs/

WORKDIR /app

COPY requirements.txt .

# Install other requirements
RUN pip install -r requirements.txt

RUN chmod -R 777 /app 

CMD [ "python", "-u", "./evaluate.py" ]
