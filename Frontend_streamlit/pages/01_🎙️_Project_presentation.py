import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import base64
import s3fs
import io

st.set_page_config(
    page_title="Hello MushroomSeeker",
    page_icon="🍄",
    layout="wide")

fs = s3fs.S3FileSystem(anon=False)
image_path = ("imagemobucket/Streamlit/Figure_project/")
df_path = ("imagemobucket/Csv/")
image_path_gif = ("imagemobucket/Streamlit/Gif/")

@st.cache_data(ttl=600)
def read_image_bucket(filename):
    return Image.open(fs.open(filename))

@st.cache_data(ttl=600)
def read_file(filename):
    with fs.open(filename, 'rb') as f:
        return f.read()


image_team_path = ("imagemobucket/Streamlit/Team/")

      
st.markdown("<h3 style='text-align: center;'>This is our team. We are glad to host you !</h3>", unsafe_allow_html=True)


st.markdown("""---""")
c1, c2, c3, c4 = st.columns(4, gap="large")


with c1 : 
    st.subheader('Joffrey Lemery')
with c2 : 
    st.subheader('Thomas Louvradoux') 
with c3 : 
    st.subheader('Julien Le Bot') 
with c4 : 
    st.subheader('Gaëtan Wendling') 
c1, c2, c3, c4 = st.columns(4, gap="large")

with c1 : 
    st.image(read_image_bucket( image_team_path + 'joffrey_lemery.jpeg'),channels="RGB", output_format="auto")
with c2 : 
    st.image(read_image_bucket( image_team_path + 'thomas_louvradoux.jpg'),channels="RGB", output_format="auto")
with c3 : 
    st.image(read_image_bucket( image_team_path + 'julien_le_bot.jpeg'),channels="RGB", output_format="auto")
with c4 : 
    st.image(read_image_bucket( image_team_path + 'gaetan_wendling.jpeg'),channels="RGB", output_format="auto")
    


c1, c2, c3, c4 = st.columns(4, gap="large")
with c1 : 
    st.write("Joffrey is a graduate engineer from France and Quebec.\nHis energy, his technical skills and his infinite thirst for knowledge bring people together around challenging projects. ")
with c2 : 
    st.write("Thomas is a PhD student in Quantum physics.\nHis sharp mind and learning methodology are essential for state-of-the-art research problems such as deep learning. ")
with c3 : 
    st.write("Julien is a technical manager.\nHis debugging skills, his analysis of priorities and his sense of management are strong points in group work. ")
with c4 : 
    st.write("Gaetan is a business genius.\nHis sense of business understanding allows him to give meaning to deep learning." )

st.markdown("""---""")

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