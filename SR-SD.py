import numpy as np
import os
import PIL
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
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import transformers
import serial
ser = serial.Serial("COM1")
transformers.logging.set_verbosity_error()
db_path = "C:\\Users\\George\\.keras\\datasets\\DB" #change and create folder before running AI
db2_path = "C:\\Users\\George\\.keras\\datasets\\MIND" #change and create folder before running AI
print("AI-Synthetic dawn");
option = input ("Do you want to: load or save the vision model. [load/save]?: ")
user_inputB = input("download or exec pretrained mind[download/exec]?:")
if user_inputB == "download":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    modelM = GPT2LMHeadModel.from_pretrained('gpt2')
    modelM.save_pretrained("./cached-GPT2")
    tokenizer.save_pretrained("./cached_t-GPT2")
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
model = Sequential([
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
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
if option == "save":    
    epochs=int(input("epochs:"))
    history = model.fit(
train_ds,
validation_data=val_ds,
epochs=epochs
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    model.save("vision_model")
if option == "load":
    model = keras.models.load_model("vision_model")
    cap = cv2.VideoCapture(0)
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
# take screenshot using pyautogui
    i = 0
    string = ""
    prev = ""
    while(True):
        sleep(1)
        i += 1
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
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
        #if new object say "what is that?"
        if 100 * np.max(score) < 50:
            print("what is that?")
            print("accepting voice input.")
            new_object = "ball" # test code for voice input
            dirname = str(new_object)
            if os.path.isdir(db_path + "\\" + new_object) == False:
                os.mkdir(db2_path + "\\" + new_object)
            cv2.imwrite(db2_path + "\\" + new_object + "\\frame_" + str(i) + ".png", image)
            #associative reasoning of new object label
        #if experiment then random actions for associative reasoning
        action = round(random.uniform(0, 10))
        print("doing random action " + str(action) + " to experiment and learn.")
        print("communicating to " + ser.portstr + " with action " + str(action))
        ser.write(str(action).encode())
        print("what am i doing?")
        print("accepting voice input to understand self agency.")
        new_agency = "pointing" # test code for voice input
        file = open('agency.txt', 'a')
        file.write(new_agency + "=" + str(action) + "\n")
        file.close()
        # search directory structure for stored successful action frame from action-vision association then do.
        for file in os.listdir(db2_path):
            if file.endswith(".png"):
                #print(os.path.join("/", file))
                #if file.find(class_names[np.argmax(score)]) == -1:
                    #print("capable of " + file + ", no motive")
                if file.find(class_names[np.argmax(score)]) != -1 and file.find("_framesuccess_") != -1:
                    processB = file.split("_")[4]
                    print("doing purposeful action " + processB)
                    #do action via serial connection to robot
                    print("communicating to " + ser.portstr + " with successful action " + processB)
                    ser.write(processB.encode())
                    with open('agency.txt') as f:
                        for line in f:
                            if line.find(file.split("_")[4]) != -1:
                                print("i am " + line.split("=")[0] + " at " + file.split("_")[0])
                                break
                    break
        if (i % 2) != 0:
            if user_inputB == "exec":
                modelM = GPT2LMHeadModel.from_pretrained('./cached-GPT2')
                tokenizer = GPT2Tokenizer.from_pretrained('./cached_t-GPT2')
                inputs = tokenizer.encode(class_names[np.argmax(score)], return_tensors='pt')
                outputs = modelM.generate(
                inputs, max_length=16, do_sample=True, temperature=5.0
                )
                string = tokenizer.decode(outputs[0], skip_special_tokens=True)
                prev = class_names[np.argmax(score)]
        if (i % 2) == 0:
            if user_inputB == "exec":
                if string.find(class_names[np.argmax(score)]) > len(prev): # '<' test mode, '>' real mode
                        print("Success in thinking pathway, stored mental frames")
                        cv2.imwrite(db2_path + "\\" + class_names[np.argmax(score)] + "_framesuccess_" + str(i) + "_action_" + str(action) + "_.png", image)
                        # action-vision association
                        process = string.split(" ")
                        #for word in process:
                            #print("downloading " + word + " images for training purposes.")
                            #download images to storage...
                        print("self improving...")
                        data_dir = pathlib.Path(db_path)
                        image_count = len(list(data_dir.glob('*/*.jpg')))
                        #print(image_count)
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
                        model = Sequential([
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
                        model.compile(optimizer='adam',
                                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
                        epochs=1
                        history = model.fit(
                        train_ds,
                        validation_data=val_ds,
                        epochs=epochs
                        )
                        acc = history.history['accuracy']
                        val_acc = history.history['val_accuracy']
                        loss = history.history['loss']
                        val_loss = history.history['val_loss']
                        model.save("improvement" + str(i))
                        model = keras.models.load_model("improvement" + str(i))
                        print("Loaded new model...")