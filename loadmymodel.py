import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("keras/finalModel")
class_names = ["bar", "beach", "clubbing", "museums", "nature", "none", "shopping"]

print(model.summary())

image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRSDvroNeJAPWuqy1oJGH3g2LqWuDXVAibuag&usqp=CAU"
image_path = tf.keras.utils.get_file("shopOtherOne", origin=image_url)

img = keras.preprocessing.image.load_img(image_path, target_size=(180, 180))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
        class_names[np.argmax(score)], 100 * np.max(score)
    )
)

