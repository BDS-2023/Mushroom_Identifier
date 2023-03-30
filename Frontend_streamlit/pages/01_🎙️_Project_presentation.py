import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import s3fs


@st.cache_data(ttl=600)
def read_image(filename):
    img =  fs.open(filename)
    return Image.open(img)

fs = s3fs.S3FileSystem(anon=False)

path_streamlit = 'imagemobucket/Streamlit/'
image_1 = 'st_image_1.jpeg'

def main():
    st.title('Generical presentation ')

    choice = st.sidebar.radio("Submenu", ["Team and project description", "Hardware and Software", "Documentation ressources"])
    if choice == 'Team and project description':
        st.subheader("This is our team. We are glad to host you !")
        st.image(read_image(path_streamlit+image_1), caption= 'Boys and the hood', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.text('The text that describe us !\nTest retour à la ligne\nGreat it works')
      

    if choice == 'Hardware and Software':
        st.subheader('Curious about the tools used in this projet ? Come have a look !')

        with st.container():
            st.subheader("Set 1 - Dev")
            df = pd.DataFrame(np.random.randn(10, 2), columns=(['Hardware or Software', 'Description']))
            st.dataframe(data=df, width=10000, height=None, use_container_width=False)

            st.subheader("Set 2 - Dev")
            df = pd.DataFrame(np.random.randn(10, 2), columns=(['Hardware or Software', 'Description']))
            st.dataframe(data=df, width=10000, height=None, use_container_width=False)

            st.subheader("Set 3 - Production")
            df = pd.DataFrame(np.random.randn(10, 2), columns=(['Hardware or Software', 'Description']))
            st.dataframe(data=df, width=10000, height=None, use_container_width=False)



    if choice == 'Documentation ressources':
        st.subheader('Please find below our selection of cool and very interesting documentation dealing with CNNs!')
        df = pd.DataFrame(np.random.randn(10, 2), columns=(['Link', 'Topic']))
        st.dataframe(data=df, width=10000, height=None, use_container_width=False)


if __name__ == '__main__':
    main()
