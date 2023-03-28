import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
import pickle
from sklearn.metrics.pairwise import cosine_similarity

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

## Function for Save the uploded Image
def save_uploded_image(uploded_image):
    try:
        with open(os.path.join('uploads',uploded_image.name), 'wb') as f:
            f.write(uploded_image.getbuffer())
        return True
    except:
        return False

# Extraxt Feature for Uploded Image
def extract_feature(img_path, model):

    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()
    return result

# Recommended Function 
def recommend(feature_list,feature):

    similarity = []

    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(feature.reshape(1,-1), feature_list[i].reshape(1,-1))[0][0])

        index_pos = sorted(list(enumerate(similarity)), reverse=True ,key= lambda x: x[1])[0][0]

    return index_pos


st.title('BIRD VS DRONE')

uploded_image = st.file_uploader('Choose an Image')

if uploded_image is not None:
    # save the image
    if save_uploded_image(uploded_image):
        #load the image
        display_image = Image.open(uploded_image)
        #Featue Extractor
        features = extract_feature( os.path.join('uploads',uploded_image.name),model)
        
        #recommend
        index_pos = recommend(feature_list,features)

        #display
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Your Uploaded Image')
            st.image(display_image)

        with col2:
            st.subheader('It is a'+ ' ' +str(filenames[index_pos].split('\\')[1])+ ' '+ 'and more similar image is:')
            #st.image('filenames[index_pos]')

