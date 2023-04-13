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
    page_title="Hello MushroomSeeker",
    page_icon="🍄",
    layout="wide")


def main():


    c1,c2,c3 = st.columns([0.5,1,0.5])
    with c2: 

        st.title('Welcome to :red[Mushroom Identifier] 👋')
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
            st.subheader("**Feel free to explore and use our CNN to identify your mushrooms. We're really proud of this realisation.**")
            st.write(' ')
            st.write(' ')
            st.subheader("Please feel free to thumbs up our [github](https://github.com/BDS-2023/Mushroom_Identifier)")
            

            st.markdown("""---""")

            col1, col2, col3= st.columns(3)

            with col2:
                file_ = (image_path_gif + "mario_team.gif")
                contents = read_file(file_) 
                data_url = base64.b64encode(contents).decode("utf-8")
                st.markdown( f'<img src="data:image/gif;base64 , {data_url}" alt="Mario gif" width="250" height="100">', unsafe_allow_html=True )

with st.sidebar:
    st.markdown("""---""")
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


if __name__ == '__main__':
    main()
       