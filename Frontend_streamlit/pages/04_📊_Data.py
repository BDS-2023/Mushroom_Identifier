import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
import folium
from streamlit_folium import st_folium,folium_static
import s3fs
import os

fs = s3fs.S3FileSystem(anon=False)


path_imgs_best  = "imagemobucket/pic_GBIF_best_of_100"
path_csvs = "imagemobucket/Csv"
image_path = ("imagemobucket/Streamlit/Figure_project/")


@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    return img


@st.cache_data
def read_file(filename):
    with fs.open(filename) as f:
        return f.read().decode("utf-8")
    
@st.cache_data
def read_csv(filename):
    return pd.read_csv(fs.open(filename))
    

@st.cache_data(ttl=600)
def read_image_bucket(filename):
    return Image.open(fs.open(filename))


df_loca_100 = read_csv(path_csvs + '/localization_top_100.csv')
df_taxon = read_csv(path_csvs + '/Taxon_100.csv')

st.set_page_config(
    page_title="Hello MushroomSeeker",
    page_icon="🍄",
    layout="wide")

st.title("Take a look at the Data")
st.header("The model is currently working with 100 class")
st.subheader("Select one of the class to see some images")

col3,col1,col2 = st.columns([1,100,1])
with col1:

    option = st.selectbox(
    'Mushroom species',
    ('Aleuria aurantia','Amanita fulva','Amanita muscaria','Amanita phalloides','Amanita rubescens','Armillaria mellea','Artomyces pyxidatus','Ascocoryne sarcoides',
 'Bjerkandera adusta','Bolbitius titubans','Boletus edulis','Byssomerulius corium','Calocera cornea','Cerioporus squamosus','Chalciporus piperatus','Chondrostereum purpureum',
 'Clavulina coralloides','Clitocybe nebularis','Clitocybe odora','Clitopilus prunulus','Coprinellus disseminatus','Coprinellus micaceus','Coprinopsis atramentaria',
 'Coprinus comatus','Cortinarius semisanguineus','Craterellus tubaeformis','Cuphophyllus pratensis','Cuphophyllus virgineus','Cystoderma amianthinum','Daedalea quercina',
 'Daedaleopsis confragosa','Exidia glandulosa','Fistulina hepatica','Flammulina velutipes','Flavoparmelia caperata','Fomes fomentarius','Fomitopsis betulina','Fomitopsis pinicola',
 'Galerina marginata','Ganoderma applanatum','Geastrum triplex','Gliophorus psittacinus','Gloeophyllum sepiarium','Grifola frondosa','Gymnopus dryophilus','Helvella crispa',
 'Hydnellum peckii','Hydnum repandum','Hygrocybe coccinea','Hygrocybe conica','Hygrophoropsis aurantiaca','Hypholoma capnoides','Hypholoma fasciculare','Imleria badia',
 'Inocybe geophylla','Laccaria amethystina','Laccaria laccata','Lacrymaria lacrymabunda','Laetiporus sulphureus','Leccinum scabrum','Lepiota cristata','Lepista nuda',
 'Lycoperdon perlatum','Macrolepiota procera','Marasmius oreades','Marasmius rotula','Mycena epipterygia','Mycena galericulata','Mycena haematopus','Mycena pura',
 'Panellus stipticus','Parmelia sulcata','Paxillus involutus','Peltigera praetextata','Phaeolus schweinitzii','Phallus impudicus','Phlebia radiata','Physcia aipolia',
 'Pleurotus ostreatus','Plicaturopsis crispa','Pluteus cervinus','Pseudohydnum gelatinosum','Rhodocollybia butyracea','Rhodocollybia maculata','Rickenella fibula',
 'Russula cyanoxantha','Sarcodon imbricatus','Schizophyllum commune','Scleroderma citrinum','Stereum hirsutum','Suillus luteus','Tapinella atrotomentosa','Trametes gibbosa',
 'Trametes ochracea','Trametes versicolor','Tremella mesenterica','Trichaptum abietinum','Tricholomopsis rutilans','Tubaria furfuracea','Tylopilus felleus'))

    
    
    st.image(read_image_bucket(path_imgs_best + '/{0}.jpg'.format(option)) ,channels="RGB", output_format="auto")

    st.markdown("""---""")
    st.subheader("Here are some places where you can find (not exclusively) some {0}".format(option))
    st.write("")
    st.write("")


loc_center = [df_loca_100['long'].mean(), df_loca_100['east'].mean()+120]


map1 = folium.Map(location = loc_center, zoom_start = 1)
for index, loc in df_loca_100[df_loca_100.real_name == option].iterrows():
    if loc["east"] > 170:
        pass
    else:
        folium.CircleMarker([loc['long'], loc['east']], popup=loc['real_name'],radius=2,weight=5).add_to(map1)
    
folium.LayerControl().add_to(map1)


st_data = folium_static(map1)

col1,col2,col3 = st.columns([1,5,1]) 
with col2:  
    taxonomy = df_taxon[df_taxon.real_name == option]
    st.subheader('Taxonomy of {0}'.format(option))
    tb = ''
    for k in range(2,9):
        st.write(tb ,'**{0}**'.format(taxonomy.iloc[:,k].name.upper()),"&nbsp;&nbsp;&nbsp;",taxonomy.iloc[:,k].values[0])
        tb += "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"

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