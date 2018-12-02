
import io
import flask
import numpy as np
from keras.models import load_model
import cv2
import tensorflow as tf

app = flask.Flask(__name__)
base_dir = "C:/Users/gabri/Google Drive"


def init_model():
    cnn_model = load_model(base_dir + '/model.h5')
    return cnn_model


global model
model = init_model()  # here load an already trained model
global graph
graph = tf.get_default_graph()


@app.route("/predict", methods=["POST", "GET"])
def predict():
    # the class for the clothing, 1 = elegant, 2 = casual for example ...
    data = {"prediction": 0}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # gets the file storage stream
            img_stream = flask.request.files['image'].read()
            np_img = np.fromstring(img_stream, np.uint8)
            cv_img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
            img = np.reshape(cv_img, [1, 32, 32, 3])
            with graph.as_default():
                data["prediction"] = np.asscalar(np.argmax(model.predict(img)))
    return flask.jsonify(data)


if __name__ == "__main__":
    print("Loading the server, please wait ...")
    app.run(debug=True)
