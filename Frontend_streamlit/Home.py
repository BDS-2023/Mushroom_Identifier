import streamlit as st
from PIL import Image
import base64
import s3fs

@st.cache_data(ttl=600)
def read_file(filename):
    with fs.open(filename, 'rb') as f:
        return f.read()

@st.cache_data(ttl=600)
def path_images(folder_name):
    files = fs.ls(folder_name)
    return files

fs = s3fs.S3FileSystem(anon=False)
path_streamlit = 'imagemobucket/Streamlit/'

def main():
    st.set_page_config(
    page_title="Hello Mushroom-dawan",
    page_icon="🍄")
    st.title('Welcome to :red[BDSMushroom] 👋')
    st.subheader('Here you are on the main page of our CNN muschroom recognition projet ! Feel free to explore and use our CNN to identify your mushrooms')
    
    col1, col2, col3  = st.columns([3,1,1])

    with col1:
        contents = read_file(path_streamlit+'Mario.gif')
        data_url = base64.b64encode(contents).decode("utf-8")
    
        st.markdown( f'<img src="data:image/gif;base64 , {data_url}" alt="Mario gif" width="700" height="500">', unsafe_allow_html=True )

if __name__ == '__main__':
    main()
