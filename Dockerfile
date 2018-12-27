FROM joelogan/keras-tensorflow-flask-uwsgi-nginx-docker
RUN pip install pandas
COPY ./app /app
