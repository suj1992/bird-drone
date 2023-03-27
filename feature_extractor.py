# Extract all the path of the images

import os
import pickle

bird_drone = os.listdir('data')

filenames = []

for bd in bird_drone:
    for file in os.listdir(os.path.join('data',bd)):
        filenames.append(os.path.join('data',bd,file))

print(filenames)
print(len(filenames))

pickle.dump(filenames, open('filenames.pkl', 'wb'))

#Create The Model For Feature Extraction

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())



filenames = pickle.load(open('filenames.pkl', 'rb'))

def feature_extractor(img_path,model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result

features = []
for file in tqdm(filenames):
    features.append(feature_extractor(file, model))

pickle.dump(features,open('embeddings.pkl','wb'))