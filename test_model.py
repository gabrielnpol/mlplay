
import numpy as np
from keras.models import load_model
import cv2

# print(train_generator.class_indices)
# {'frog': 0, 'truck': 1, 'deer': 2, 'automobile': 3, 'bird': 4, 'horse': 5, 'ship': 6, 'cat': 7, 'dog': 8, 'airplane': 9}
#model.save("/content/drive/My Drive/model.h5")
base_dir = "C:/Users/gabri/Google Drive"
model = load_model(base_dir + '/model.h5')
img = cv2.imread(base_dir + "/train_subset.zip (Unzipped Files)/train_subset/200.png")
#img = cv2.resize(img,(32,32))
img = np.reshape(img,[1,32,32,3])
print(np.argmax(model.predict(img)))