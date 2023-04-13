import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
import base64
import s3fs
import random
import os

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

@st.cache_data
def read_csv(filename):
    return pd.read_csv(fs.open(filename))

@st.cache_data(ttl=600)
def read_image_bucket(filename):
    return Image.open(fs.open(filename))

path_csvs = "imagemobucket/Csv"
df_loca_100 = read_csv(path_csvs + '/localization_top_100.csv')
image_path = ("imagemobucket/Streamlit/Figure_project/")



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

st.set_page_config(
    page_title="Hello MushroomSeeker",
    page_icon="🍄",
    layout="wide")

st.title("Mushroom's Recognition App")
st.subheader("Yes, it is the core of the projet. Give it a try !")


st.subheader("Time for you to use our App and to answer you question : What kind of mushroom did I found with grand'dad ?")
image = st.file_uploader('Upload your Mushroom picture', type=['jpg', 'jpeg'], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
columns = st.columns((2, 1, 2))
button = columns[1].button("Let's recognize", help='Click to start recognition', on_click=None, args=None, kwargs=None, type="primary", disabled=False)
if button :
    if image is not None :
        files = {"file": image.getvalue()}
        requestpost = requests.post(f"http://155.155.0.214:8000/uploadfile", files=files)
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
        col1, col2, col3, col4, col5 = st.columns([1,4,1,4,1], gap = 'small')
        with col2:
            if image is not None :
                st.image(load_image(image), caption= 'To be predicted', width=None, use_column_width = 'auto', clamp=False, channels="RGB", output_format="auto")
        with col4:
            if image is not None :
                try:
                    st.image(decoded_image, caption= "{0}: {1}%".format(name_top_5[0],preds[0]), width=None, use_column_width='auto', clamp=False, channels="RGB", output_format="auto")
                except:
                    pass

if button :
    with st.container():
        st.text('On the right side is represented the GradCam of the model Analysis.\nThis methods aims to explain the interpratbility of model by highliting\nthe most important zones of the picture taking part in the classification')
        
if button :
    st.subheader('Taxonomy of {0}'.format(name_top_5[0]))
    col1, col2, col3= st.columns([1,4,1], gap = 'large')
    with col2:
        df_taxon = read_csv(path_csvs + '/Taxon_100.csv')
        taxonomy = df_taxon[df_taxon.real_name == name_top_5[0]]
        tb = ''
        for k in range(2,9):
            st.write(tb ,'**{0}**'.format(taxonomy.iloc[:,k].name.upper()),"&nbsp;&nbsp;&nbsp;",taxonomy.iloc[:,k].values[0])
            tb += "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"



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
            st.subheader("Pictures from the same species")
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

with st.sidebar:
    with st.expander("Joffrey Lemery"):
        col1, col2, col3 = st.columns([1,0.5,1])  
        with col1: 
            st.image(read_image_bucket( image_path + 'LinkedIn_Logo_blank.png'),channels="RGB", output_format="auto") 
            st.image(read_image_bucket( image_path + 'github_blank.png'),channels="RGB", output_format="auto")
        with col3:
            st.write("[Linkedin](https://www.linkedin.com/in/joffrey-lemery-b740a5112/)")
            st.write("")
            st.write("")
            st.write("[GitHub](https://github.com/JoffreyLemery)")
            
    
    with st.expander("Thomas Louvradoux"):
        col1, col2, col3 = st.columns([1,0.5,1])  
        with col1: 
            st.image(read_image_bucket( image_path + 'LinkedIn_Logo_blank.png'),channels="RGB", output_format="auto") 
            st.image(read_image_bucket( image_path + 'github_blank.png'),channels="RGB", output_format="auto")
        with col3:
            st.write("[Linkedin](https://www.linkedin.com/in/thomas-louvradoux-023b231a6/)")
            st.write("")
            st.write("")
            st.write("[GitHub](https://github.com/Louvradoux)")

    with st.expander("Julien Le Bot"):
        col1, col2, col3 = st.columns([1,0.5,1])  
        with col1: 
            st.image(read_image_bucket( image_path + 'LinkedIn_Logo_blank.png'),channels="RGB", output_format="auto") 
            st.image(read_image_bucket( image_path + 'github_blank.png'),channels="RGB", output_format="auto")
        with col3:
            st.write("[Linkedin](https://www.linkedin.com/in/julien-le-bot-133a5625//)")
            st.write("")
            st.write("")
            st.write("[GitHub](https://github.com/jlebot44)")

    with st.expander("Gaëtan Wendling"):
        col1, col2, col3 = st.columns([1,0.5,1])  
        with col1: 
            st.image(read_image_bucket( image_path + 'LinkedIn_Logo_blank.png'),channels="RGB", output_format="auto") 
            st.image(read_image_bucket( image_path + 'github_blank.png'),channels="RGB", output_format="auto")
        with col3:
            st.write("[Linkedin](https://www.linkedin.com/in/gaetan-wendling/)")
            st.write("")
            st.write("")
            st.write("[GitHub](https://github.com/GaetanWendling)")