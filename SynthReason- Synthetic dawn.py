import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import transformers
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import cv2
import pyautogui
import webbrowser
import random
from time import sleep
from google_images_search import GoogleImagesSearch
transformers.logging.set_verbosity_error()
print("AI-Synthetic dawn");

user_input = input("download or exec pretrained language model[download/exec]?:")
file = open('questions.txt', encoding="utf8").read().splitlines()
file_object = open('output.txt', 'a', encoding="utf-8")
if user_input == "download":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.save_pretrained("./cached-GPT2")
    tokenizer.save_pretrained("./cached_t-GPT2")
if user_input == "exec":
    model = GPT2LMHeadModel.from_pretrained('./cached-GPT2')
    tokenizer = GPT2Tokenizer.from_pretrained('./cached_t-GPT2')
db_path = "C:\\Users\\George\\.keras\\datasets\\DB" #change and create folder before running AI
db2_path = "C:\\Users\\George\\.keras\\datasets\\MIND" #change and create folder before running AI
option = input ("Do you want to: load the vision model or train the vision model [load/train]?:")
optionB = input ("Do you want to SynthReason to learn [yes/no]?:")
if optionB == "yes":
    imageDLCount = 10
    for x in range(imageDLCount):
        # you can provide API key and CX using arguments,
        # or you can set environment variables: GCS_DEVELOPER_KEY, GCS_CX
        gis = GoogleImagesSearch('AIzaSyA7LMY8kcMeOeQyxV_5RIjuU01kdBgOZbY', '349e6227511e8004c')
        string = random.choice(open("objects.txt").readlines()).rstrip('\n')
        print("Downloading " + string + ".")
        _search_params = {
            'q': string,
            'num': 10,
            'fileType': 'jpg',
            'rights': 'cc_publicdomain',
            'safe': 'off', ##
            'imgType': 'photo', ##
            'imgSize': 'imgSizeUndefined', ##
            'imgDominantColor': 'imgDominantColorUndefined', ##
            'imgColorType': 'color' ##
        }
        path = db_path + '\\' + string + '\\'
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        gis.search(search_params=_search_params, path_to_dir=path, width=500, height=500)
        #for image in gis.results():
           # image.url  # image direct url
            #image.referrer_url  # image referrer url (source) 
           # image.download(path)  # download image
           # image.resize(500, 500)  # resize downloaded image
           # image.path  # downloaded local file path
        print("Done.")
data_dir = pathlib.Path(db_path)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
batch_size = 32
img_height = 180
img_width = 180
train_ds = tf.keras.utils.image_dataset_from_directory(
data_dir,
validation_split=0.2,
subset="training",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
data_dir,
validation_split=0.2,
subset="validation",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
num_classes = len(class_names)
modelV = Sequential([
layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
layers.Conv2D(16, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(32, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(64, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dense(num_classes)
])
modelV.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
if option == "train":    
    epochs=int(input("epochs:"))
    history = modelV.fit(
train_ds,
validation_data=val_ds,
epochs=epochs
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    modelV.save("vision_model")
if option == "load":
    modelV = keras.models.load_model("vision_model")
    cap = cv2.VideoCapture(0)
while(True):
        print("---------------------------------------")
        #webcam stuff
#   while(cap.isOpened()):
#        ret, frame = cap.read()
#        if ret == False:
#            break
#        cv2.imwrite('Frame'+str(i)+'.jpg', frame)
#        sunflower_path = 'Frame'+str(i)+'.jpg'
#        img = tf.keras.utils.load_img(
#        sunflower_path, target_size=(img_height, img_width)
#        )
#        img_array = tf.keras.utils.img_to_array(img)
#        img_array = tf.expand_dims(img_array, 0) # Create a batch
#        predictions = model.predict(img_array)
#        score = tf.nn.softmax(predictions[0])
#        print(
#        "This image most likely belongs to {} with a {:.2f} percent confidence."
#        .format(class_names[np.argmax(score)], 100 * np.max(score))
#        )
#        i += 1
#cap.release()
#cv2.destroyAllWindows()
        image = pyautogui.screenshot()
        image = cv2.cvtColor(np.array(image),
                     cv2.COLOR_RGB2BGR)
        cv2.imwrite(db2_path + "\\frame.png", image)
        sunflower_path = db2_path + "\\frame.png"
        img = tf.keras.utils.load_img(
        sunflower_path, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = modelV.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
        print()
        prev = class_names[np.argmax(score)]
        action = round(random.uniform(0, 10))#do random action or load action from agency
        new_agency = ""
        with open('agency.txt') as f:
            for line in list(f):
                if line.find(class_names[np.argmax(score)]) != -1:
                    arr = line.split("=")
                    print("AI: i am doing action " + arr[1] + " with " + arr[0])
                    action = line.split("=")[1]
                    new_agency = action
                    break
        with open('agency.txt') as f:
            check = class_names[np.argmax(score)] + "=" + str(new_agency)
            if check not in f.read():
                proc = "i am doing random action " + str(action) + " with " +  class_names[np.argmax(score)]
                print(proc.rstrip())
        inputs = tokenizer.encode(class_names[np.argmax(score)], return_tensors='pt')
        n = 0
        attempts = 5
        mindSize = 5
        while(True):
            m = 0
            x = range(mindSize)
            text = ""
            for m in x:
                outputs = model.generate(
                inputs, max_length=64, do_sample=True, temperature=7.0
                )
                text += tokenizer.decode(outputs[0], skip_special_tokens=True)[len(class_names[np.argmax(score)]):]
                m += 1
            print()
            print("AI: " + text)
            print()
            image = pyautogui.screenshot()
            image = cv2.cvtColor(np.array(image),
                        cv2.COLOR_RGB2BGR)
            cv2.imwrite(db2_path+"\\frame.png", image)
            sunflower_path =db2_path + "\\frame.png"
            img = tf.keras.utils.load_img(
            sunflower_path, target_size=(img_height, img_width)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            predictions = modelV.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            print(
            "The next image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
            )
            if text.replace(prev,"").find(class_names[np.argmax(score)]) == -1:
                n += 1
            if n > attempts:
                break
            if text.replace(prev,"").find(class_names[np.argmax(score)]) > -1:
                n = 0
                with open('agency.txt') as f:
                    check = class_names[np.argmax(score)] + "=" + str(action)
                    if check in f.read():
                        print("AI: done action " + str(action) + " with " + prev + " now I see " + class_names[np.argmax(score)])
                    else:
                        print("AI: recording successful action")
                        file = open('agency.txt', 'a')
                        file.write(class_names[np.argmax(score)] + "=" + str(new_agency) + "\n")
                        file.close()
                print()
                break
#TODO add webcam support
#associate words with actions by narrowing down specific verbs within mutiple action attempts, self awareness, then the AI is able to rationalise motor control by selecting how to manipulate the environment to obtain the experience of a particular object...
#add serial connection to motor control
#add research mode for new objects, training of new vision model
#resolve timing issue
#3D limb dependent cartesian motor control
