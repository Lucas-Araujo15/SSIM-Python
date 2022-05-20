import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
from scipy.spatial import distance

model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

IMAGE_SHAPE = (224, 224)

layer = hub.KerasLayer(model_url)
model = tf.keras.Sequential([layer])


def extract(file):
    file = Image.open(file).convert('L').resize(IMAGE_SHAPE)
    # display(file)

    file = np.stack((file,)*3, axis=-1)

    file = np.array(file)/255.0

    embedding = model.predict(file[np.newaxis, ...])
    # print(embedding)
    vgg16_feature_np = np.array(embedding)
    flattended_feature = vgg16_feature_np.flatten()

    # print(len(flattended_feature))
    print(flattended_feature)
    # print('-----------')
    return flattended_feature

img1 = extract('images/2.jpg')
img2 = extract('images/3.jpg')
img3 = extract('images/senna1.jpg')

metric = 'cosine'

dc = distance.cdist([img1], [img2], metric)[0]
print(dc)
print("the distance between original and the original is {}".format(dc))
