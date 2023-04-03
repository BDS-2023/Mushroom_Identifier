import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import base64
import s3fs
import io

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


st.set_page_config(
    page_title="Hello Mushroom-dawan",
    page_icon="🍄",
    layout="wide")

        
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
    st.write("Joffrey est l'inventeur du VVG20.\nKeras n'a plus de secret pour lui.\nOn est jaloux de son abondance \nd'énergie. ")
with c2 : 
    st.write("Thomas à connu Albert Einstein.\nIl s'est fait voler E=MC2.\nFastAI est une formalité pour lui. ")
with c3 : 
    st.write("Julien est un debugger hors normes.\nRédacteur en chef du rapport projet.\nHabitant de Bretagne il sait quand il \npleut avant tout le monde. ")
with c4 : 
    st.write("Gaëtan Overflood les channels de GIF.\nC'est lui qui à ecrit ces bêtises.\nEst en train d'essayer de faire un site\njolie." )

st.markdown("""---""")
