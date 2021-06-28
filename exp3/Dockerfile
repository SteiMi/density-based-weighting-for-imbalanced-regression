FROM INSERT_YOUR_REGISTRY/deepsd-requirements:latest

COPY *.py run_job.sh /app/
COPY ./srcnn-tensorflow /app/srcnn-tensorflow
COPY data /app/data

WORKDIR /app

RUN chmod -R 777 /app 

CMD [ "sh", "-c", "./run_job.sh" ]
