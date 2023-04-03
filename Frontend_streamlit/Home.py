import streamlit as st
from PIL import Image
import base64
from streamlit.components.v1 import html
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

st.set_page_config(
    page_title="Hello Mushroom-dawan",
    page_icon="🍄",
    layout="wide")


def main():


    c1,c2,c3 = st.columns([0.5,1,0.5])
    with c2: 

        st.title('Welcome to :red[BDSMushroom] 👋')
        st.title(" ")

    with st.container():
        col1, col2= st.columns([1,1])

        with col1:
            image_path = ("imagemobucket/Streamlit/Figure_project/")
            st.image(read_image_bucket( image_path + 'word_mush.png'),channels="RGB", output_format="auto", use_column_width=True)

        with col2:
            st.markdown("""---""")
            st.subheader("Here you are on the main page of our CNN muschroom recognition projet !")


            st.write("") 
            st.subheader("**Feel free to explore and use our CNN to identify your mushrooms we're really proud of this realisation.**")
            st.write(' ')
            st.write(' ')
            st.subheader("Please feel free to thumbs up our [github](https://github.com/BDS-2023/Mushroom_recognition)")
            

            st.markdown("""---""")

            col1, col2, col3= st.columns(3)

            with col2:
                file_ = (image_path_gif + "mario_team.gif")
                contents = read_file(file_) 
                data_url = base64.b64encode(contents).decode("utf-8")
                st.markdown( f'<img src="data:image/gif;base64 , {data_url}" alt="Mario gif" width="250" height="100">', unsafe_allow_html=True )

if __name__ == '__main__':
    main()
       