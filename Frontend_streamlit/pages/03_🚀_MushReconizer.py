import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
import base64
import s3fs
import random

@st.cache_data(ttl=600)
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Create connection object.
# `anon=False` means not anonymous, i.e. it uses access keys to pull data.
fs = s3fs.S3FileSystem(anon=False)

# Retrieve file contents.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def read_file(filename):
    with fs.open(filename) as f:
        return f.read().decode("utf-8")

@st.cache_data(ttl=600)
def read_image(filename):
    with fs.open(filename) as f:
        return Image.open(f).resize((224,224))
    
@st.cache_data(ttl=600)
def path_images(folder_name):
    files = fs.ls(folder_name)
    return files
    

def main():
    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #000099;
        color:#ffffff;
    }
    div.stButton > button:hover {
        background-color: #00FF00;
        color:#ff0000;
        }
    </style>""", unsafe_allow_html=True)

    path_image = 'imagemobucket/pic_GBIF_100_300/test/'

    st.title("Mushroom's Recognition App")
    st.subheader("Yes, it is the core of the projet. Give it a try !")
    choice = st.sidebar.radio("Submenu", ["Visual recognition", "Interpretability", "Where do I find my mushroom ?"])
    
    
    if choice == 'Visual recognition':
        st.subheader("Time for you to use our App and to answer you question : What kind of mushroom did I found with grand'dad !")
        image = st.file_uploader('Upload your Mushroom picture', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        columns = st.columns((2, 1, 2))
        button = columns[1].button("Let's recognize", help='Click to start recognition', on_click=None, args=None, kwargs=None, type="primary", disabled=False)
        if button :
            if image is not None :
                files = {"file": image.getvalue()}
                requestpost = requests.post(f"http://localhost:8000/uploadfile", files=files)
                st.text(requestpost.status_code)
                response_data = requestpost.json()

                res_top_5 = response_data.get("content")
                grad  = response_data.get("grad-cam")
                decoded_image= base64.b64decode(grad)
                ress = res_top_5.split(',')
                name_top_5 = [truc.split(':')[0] for truc in ress]
                preds = [truc.split(':')[1] for truc in ress]
        if button :
            with st.container():
                st.subheader("Prediction and interpretability")
                col1, col2 = st.columns([1,1], gap = 'large')
                with col1:
                    if image is not None :
                        st.image(load_image(image), caption= 'To be predicted', width=None, clamp=False, channels="RGB", output_format="auto")
                with col2:
                    if image is not None :
                        try:
                            st.image(decoded_image, caption= "{0}: {1}%".format(name_top_5[0],preds[0]), width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
                        except:
                            pass
        if button :
            with st.container():
                st.text('On the right side is represented the GradCam of the model Analysis.\nThis methods aims to explain the interpratbility of model by highliting\nthe most important zones of the picture taking part in the classification')
        
        if button :
            with st.container():
                st.subheader("Predictions details")
                try :
                    df = pd.DataFrame(columns=['Label', 'Probability'])
                    df['Label'] = name_top_5
                    df['Probability'] = preds
                    st.dataframe(data = df, width=10000, height=None, use_container_width=False)
                except :
                    df = pd.DataFrame(np.zeros((5,2)), columns=['Label', 'Probability'])
                    st.dataframe(data = df, width=10000, height=None, use_container_width=False)
    
        
            
            if button :
                with st.container():
                    st.subheader("Pictures from the same family")
                    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1], gap = 'small')
                    folder = path_image + df['Label'][0]+'/'
                    files = path_images(folder)
                    liste_image = random.sample(range(0,len(files)), 5)
                    try:
                        with col1:
                            if image is not None :
                                st.image(read_image(files[liste_image[0]]), width=None, clamp=False, channels="RGB", output_format="auto")
                        with col2:
                            if image is not None :
                                st.image(read_image(files[liste_image[1]]), width=None, clamp=False, channels="RGB", output_format="auto")
                        with col3:
                            if image is not None :
                                st.image(read_image(files[liste_image[2]]), width=None, clamp=False, channels="RGB", output_format="auto")
                        with col4:
                            if image is not None :
                                st.image(read_image(files[liste_image[3]]), width=None, clamp=False, channels="RGB", output_format="auto")
                        with col5:
                            if image is not None :
                                st.image(read_image(files[liste_image[4]]), width=None, clamp=False, channels="RGB", output_format="auto")
                    except: pass

    
    if choice == 'Interpretability':
        st.subheader('Have in look in the box, and check the main characteristic of the picture for the classification !')
        st.text("Interpretation conceptual presentation and invitation to click on the button if the user wants to make a try")
        columns_1 = st.columns((1, 1, 1))
        original_picture = columns_1[1].image("https://static.streamlit.io/examples/dog.jpg", caption= 'Identified mushroom', width=None, use_column_width='auto', clamp=False, channels="RGB", output_format="auto")
        columns_2 = st.columns((2, 1, 2))
        button_pressed = columns_2[1].button("Let's analyse", help='Click to start recognition', on_click=None, args=None, kwargs=None, type="primary", disabled=False, use_container_width=False)
        st.subheader('Interpretability Analisys!')
        col1, col2, col3 = st.columns([1,1,1], gap = 'medium')
        col1.image("https://static.streamlit.io/examples/dog.jpg", caption= 'Analyse 1', width=None, use_column_width='auto', clamp=False, channels="RGB", output_format="auto")
        col2.image("https://static.streamlit.io/examples/dog.jpg", caption= 'Analyse 2', width=None, use_column_width='auto', clamp=False, channels="RGB", output_format="auto")
        col3.image("https://static.streamlit.io/examples/dog.jpg", caption= 'Analyse 3', width=None, use_column_width='auto', clamp=False, channels="RGB", output_format="auto")
    
    
    
    if choice == 'Where do I find my mushroom ?':
        st.subheader("Curious about other places you could find this Mushroom ? Let's check our MushMap")



if __name__ == '__main__':
    main()