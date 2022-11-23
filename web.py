import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np


st.set_page_config(page_title='Timepass',page_icon='👽')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
	content:'Made with ❤️ by team NORA'; 
	visibility: visible ;
}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

model = load_model('saved_model.h5')

categories = ['Monkeypox', 'Measles', 'Normal', 'Chickenpox']

string1 = '<img src="https://i.postimg.cc/XYyGMJRR/Add-a-little-bit-of-body-text.png" height="120" width="700">'

st.markdown(string1, unsafe_allow_html=True)

st.write(" ")
st.write(" ")
st.write(" ")

image_file = st.file_uploader('Upload Image',type=["png","jpg","jpeg"])

if image_file:
    st.image(image_file, use_column_width=True)
    img = image.load_img(image_file,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y = model.predict(x)
    y = categories[np.argmax(y)]
    st.error(y)