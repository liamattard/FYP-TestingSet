import os
import json
import requests
import webbrowser
import urllib.request
import torch
import csv
import numpy as np
import torchvision.models as models
import tensorflow as tf
from tensorflow import keras


from PIL import Image
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F
from urllib.request import urlopen
from csv import reader
from os import listdir
from os.path import isfile, join

valid_clubbing = []
valid_bars = []
valid_beach = []
valid_nature = []
valid_shopping = []
valid_museums = []


def scene_detection(img_url, arch):

    # load the pre-trained weights
    model_file = "./%s_places365.pth.tar" % arch

    model = models.__dict__[arch](num_classes=365)

    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)

    state_dict = {
        str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    # load the image transformer
    centre_crop = trn.Compose(
        [
            trn.Resize((256, 256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(urlopen(img_url))
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    result = idx[0].item()

    if result in valid_bars:
        return "bar"

    elif result in valid_beach:
        return "beach"

    elif result in valid_clubbing:
        return "clubbing"

    elif result in valid_museums:
        return "museums"

    elif result in valid_nature:
        return "nature"

    elif result in valid_shopping:
        return "shopping"

    else:
        return "none"


model = keras.models.load_model("keras/finalModel")

class_names = ["bar", "beach", "clubbing", "museums", "nature", "none", "shopping"]


def detectWithKeras(url, count):
    image_url = url
    image_path = tf.keras.utils.get_file((str(count) + "bar"), origin=image_url)

    img = keras.preprocessing.image.load_img(image_path, target_size=(180, 180))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return class_names[np.argmax(score)]


with open("image_classificationP2.csv", "r") as read_obj:
    csv_reader = reader(read_obj)
    for i in csv_reader:
        if i[1] == "Clubbing":
            valid_clubbing.append(int(i[0]))

        if i[1] == "Bar":
            valid_bars.append(int(i[0]))

        if i[1] == "Museums":
            valid_museums.append(int(i[0]))

        if i[1] == "Nature":
            valid_nature.append(int(i[0]))

        if i[1] == "Shopping":
            valid_shopping.append(int(i[0]))

        if i[1] == "Beach":
            valid_beach.append(int(i[0]))


# Get all of the photos
folders = [name for name in os.listdir(".") if os.path.isdir(os.path.join(".", name))]

count = 0

with open("barResult.csv", "w", encoding="UTF8") as f:

    writer = csv.writer(f)

    for folder in folders:
        print("CURRENT FOLDER: " + folder)

        if folder == "bar":

            files = [f for f in listdir(folder) if isfile(join(folder, f))]

            print("current category:" + folder)

            for file in files:

                # print("current file:" + file)
                f = open((folder + "/" + file),)
                current_file = json.load(f)

                for photo in current_file["results"]:
                    url = photo["urls"]["regular"]
                    resnet18 = scene_detection(url, "resnet18")
                    print("resnet 18 detected" + resnet18)
                    resnet50 = scene_detection(url, "resnet50")
                    print("resnet 50 detected" + resnet50)
                    kerasModel = detectWithKeras(url, count)
                    print("Keras detected: " + kerasModel)

                    writer.writerow(["resnet18", folder, resnet18])
                    writer.writerow(["resnet50", folder, resnet50])
                    writer.writerow(["keras", folder, kerasModel])
                    count += 1


print(count)

