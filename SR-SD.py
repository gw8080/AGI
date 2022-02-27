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
transformers.logging.set_verbosity_error()
print("AI-Synthetic dawn");
option = input ("Do you want to: load or save the model. [load/save]? : ")
user_inputB = input("download or exec pretrained mind[download/exec]?:")

if user_inputB == "download":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.save_pretrained("./cached-GPT2")
    tokenizer.save_pretrained("./cached_t-GPT2")
data_dir = pathlib.Path("C:\\Users\\George\\.keras\\datasets\\DB")
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
    model.save("my_model")
if option == "load":
    model = keras.models.load_model("my_model")
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
    file_object = open('temp.txt', 'w')
    string = ""
    while(True):
        i += 1
        image = pyautogui.screenshot()
        image = cv2.cvtColor(np.array(image),
                     cv2.COLOR_RGB2BGR)
        cv2.imwrite("frame.png", image)
        sunflower_path = 'Frame.png'
        img = tf.keras.utils.load_img(
        sunflower_path, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
        time.sleep(1)
        if (i % 2) != 0:
            if user_inputB == "exec":
                model = GPT2LMHeadModel.from_pretrained('./cached-GPT2')
                tokenizer = GPT2Tokenizer.from_pretrained('./cached_t-GPT2')
                inputs = tokenizer.encode(class_names[np.argmax(score)], return_tensors='pt')
                string = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if (i % 2) == 0:
            if user_inputB == "exec":
                if string.find(class_names[np.argmax(score)]) != -1:
                        print("Success")
                        break