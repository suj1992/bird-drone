import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

#load Image

img = image.load_img('sample/3.jpg', target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img)

result = model.predict(preprocessed_img).flatten()

similarity = []

for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1), feature_list[i].reshape(1,-1))[0][0])

index_pos = sorted(list(enumerate(similarity)), reverse=True ,key= lambda x: x[1])[0][0]


temp_img = cv2.imread(filenames[index_pos])
cv2.imshow('output', temp_img)
cv2.waitKey(0)