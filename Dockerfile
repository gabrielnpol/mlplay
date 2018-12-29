FROM joelogan/keras-tensorflow-flask-uwsgi-nginx-docker
RUN pip install keras
RUN pip install pandas
RUN pip install opencv-python
COPY ./app /app
