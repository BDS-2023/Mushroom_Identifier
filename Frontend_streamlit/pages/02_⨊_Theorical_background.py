import streamlit as st
import base64
from PIL import Image
import pandas as pd
import numpy as np
import s3fs
import plotly.express as px
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

@st.cache_data
def read_csv_st(filename , sep_st):
        return pd.read_csv(fs.open(filename), sep= sep_st)
path = (df_path + 'GBIF_tax.csv')
df = read_csv_st(path ,sep_st='\t')
df2 = read_csv_st(df_path + "All_ranks_and_occurence.csv",sep_st=',')




choice = st.sidebar.radio("Submenu", ["Introduction","Data", "Virtual Machines", "Classification", "Interpretability"])
if choice == 'Introduction':
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title('Project report')
        c1, c2, c3, c4 = st.columns(4, gap="large")
    st.markdown("""---""")


    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:   
        st.title('Context')
        st.markdown("---")
        st.write("""There are different techniques to identify mushroom species. The most used and oldest is the morphological identification, which classifies individuals by their anatomical characteristics. However, this technique has the disadvantage of being limited in terms of accuracy because it depends on the observation and the identification protocol of the person who performs it.""")
        st.write("")
        st.write("")
        st.write("")
        st.write("""The objective of the "Mushroom Recognition" project was to perform automated recognition of mushrooms using Computer Vision technologies. It was part of the Data Scientist training provided by the company DataScientest and was a first professional experience for us in order to enhance our knowledge and increase our competence in the field of Data Science. More precisely, this project was in the field of computer vision and had for objective to acquire knowledge on Deep Learning technologies.""")
        
if choice == 'Data':
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title('Project report')
        c1, c2, c3, c4 = st.columns(4, gap="large")
    st.markdown("""---""")



    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:   
        st.title('Data')
        st.write("")
        st.write("""For the project we worked with images downloaded from the databases available on the site https://www.gbif.org/. GBIF is a scientific project aiming at indexing all taxonomic data on biodiversity, including the Fungi Kingdom. The GBIF portal exposes an API allowing to download photos of living species from a taxonomic key. There is, on the GBIF site, a file that allows to link taxonomic keys to species names and their complete taxonomies. Here are some lines from the GBIF_tax.csv file for the entries belonging to the Kingdom Fungi.""")
        st.write("")
        st.write("")
        st.write("")
        expander = st.expander("Mushrooms DataFrame")
        
        expander.dataframe(df)
        expander.write("""The column numberOfOccurences indicates the number of images available for each species. In total the source exposes about 34 M pictures of mushrooms.""")
    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2: 
        st.title("Data Exploration")
        st.write("")
        st.write("""The first step was to define our dataset. It was necessary to define the classes to use to train a model. A picture represents a mushroom which is defined by its name and its taxonomy. In biology, a taxonomy is a way to define hierarchical groups based on common characteristics (see figure).  The further down a taxonomic tree you go, the more specific the identification becomes.""")
        st.write("")
        st.write("")
        st.write("")
    col5, col6, col7, col8 = st.columns([0.5,4,4,0.5])
    with col6 : 
        st.image(read_image_bucket( image_path + 'tax_black.png'),channels="RGB", output_format="auto")
    st.markdown("""---""")
    with col7: 
        st.write("""In the table we find the column taxonRank which indicates the last level reached in the identification of the fungi pictures for a certain taxonKey. At each taxon rank (e.g. Family) we find a certain number of classes, a number that increases exponentially as we go down the tree. The figure below is a sunburst representing all the taxonomic ranks (starting from Kingdom Fungi) as well as all the classes in each rank.""")
    st.write("")
    st.write("")

    df4 = df.dropna(subset = ['kingdom', 'phylum', 'class','order','family','genus'])
    fig2 = px.sunburst(df4, path=['kingdom','phylum','class','order','family','genus'],values=df4.numberOfOccurrences,maxdepth=3,color_discrete_sequence=px.colors.qualitative.T10)
    c1,c2,c3 = st.columns([0.2,4,0.2])
    with c2: 
        st.plotly_chart(fig2 ,width= 900) 
        
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:
        

        st.write("The figure below represents the number of classes available in each taxonomic rank.")
              
    c1,c2,c3 = st.columns([0.5,1,0.5])
    with c2:
        df_forbar = df2.groupby("rank").count().sort_values(by='occurence')
        df_forbar['rank'] = df_forbar.index.values
        fig3 = px.bar(data_frame=df_forbar,x='rank',y='occurence',barmode='group',log_y=True,hover_data=["occurence"],
             labels={'rank':'Taxonomic rank','occurence':'Number of taxa'},title="Number of taxa at each taxonomic rank")
        st.plotly_chart(fig3)

        st.write("")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:
        st.write("""These figures highlight one of the main problems of the project: the huge number of different mushroom species. Indeed, on GBIF, we find 117 539 species of mushrooms. A study of the state of the art on classification by Computer Vision shows that a model capable of identifying so many classes is currently out of reach technologically speaking.""")
    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:  
        st.write("")
        st.write("")
        st.write("""In order to reduce the number of classes available, it may seem wise to move up the taxonomic tree. For example, there are only 7 different Phyla for fungi. However, by working high in the taxonomic tree, we end up with very general classes, in which the fungi can be very different. An example is given in the figure below where we present two fungi belonging to the same Family: Agaricaceae.""")
    st.write("")
    st.write("")
    col5, col6, col7 = st.columns([0.5,8,0.5])
    with col6 : 
        st.image(read_image_bucket( image_path + 'Agaricaceae.png'),channels="RGB", output_format="auto")
    
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:
        st.write("""We felt that working with such data could make learning very difficult, as we are asking the model to group very different images into the same class.""")
        st.write("")
        st.write("")
        st.write("""Finally, we decided to work on the Species mesh in order to have, in the same class, mushrooms that are similar, even if this leads to an increase in the number of available classes. For the number of classes, we set ourselves the goal of working with the 100 most represented classes in GBIF. The figure below shows the number of images available for each of these classes.""")
    st.write("")
    st.write("")
    
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:
        st.write("""Here we see that the number of images available will not be a problem. It is even necessary to restrict ourselves in order to be able to train models on our machines in acceptable times. We have therefore chosen to work with about 500 images per species, i.e. a total of 50,000 images.""")
    
    dft = df2 
    dft_temp = dft[dft['rank'] == 'Species'].head(100)
    fig4 = px.bar(data_frame=dft_temp,x='name',y='occurence',barmode='group',log_y=False,hover_data=["occurence"],color="occurence",
             labels={'name':'Taxa','occurence':"Number of image"},title="Number of images available for the first 100 taxa of the Species rank", height=800, width=1400)
    st.plotly_chart(fig4)


if choice == 'Virtual Machines':
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title('Project report')
    st.markdown("""---""")
    expander3 = st.expander("Within the framework of our project, here is the virtual machine model that we wanted to develop:") 
    with st.expander("Within the framework of our project, here is the virtual machine model that we wanted to develop:") :
        col1, col2, col3 = st.columns([0.5,2,0.5])
        with col2:
            st.image(read_image_bucket( image_path + 'DE_big.png'),channels="RGB", output_format="auto", use_column_width = 'auto')
            """
* Frontend : 
  * Streamlit allowing the reception of image classifier
    * Host = EC2 Public
* Backend : 
  * Classification of photos received by the Streamlit and data processing
    * Host = EC2 Private 
        * Downloading images from the MO database
        * Downloading images from the GBIF database
        * Update of the data received on the MO base
        * Prediction of a taxonomy
    * Host = EC2 Private Nvidia
        * NVIDIA prediction model calculations
"""

    st.markdown("""---""")

    expander3 = st.expander("To optimize our time on the models, we finally performed this architecture:") 
    with st.expander("To optimize our time on the models, we finally performed this architecture:") :
        col1, col2, col3 = st.columns([0.5,2,0.5])
        with col2:
            st.image(read_image_bucket( image_path + 'DE_small.png'),channels="RGB", output_format="auto", use_column_width = 'auto')
            """
* Frontend : 
  * Streamlit allowing the reception of image classifier
    * Host = EC2 Public
* Backend : 
  * Classification of photos received by the Streamlit and data processing
    * Host = EC2 Private
        * Prediction of a taxonomy
"""


if choice == 'Classification':
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title('Project report')
        c1, c2, c3, c4 = st.columns(4, gap="large")
    st.markdown("""---""")



    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:   
        st.title('Methods')
        st.write("")
        st.write("""As mentioned before, this project aims at identifying images of mushrooms. This process is a typical image classification problem, usually treated by Deep Learning methods and more specifically by convolutional neural networks (CNN). Moreover, it is nowadays usual to use transfer learning methods in order to reduce considerably the training time of the models.""")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("""The principle is simple, we use the knowledge acquired by models trained on very powerful machines and for long times for a particular task to solve a different problem but with similarities. For CNNs, there are now many models (VGG, MobileNet, ResNet, DenseNet, etc.) that have been trained on the ImageNet dataset, consisting of 1000 classes and 1281,167 images. The principle of transfer learning (shown in the figure below) is to interface the convolutional part of the so-called pre-trained models as well as their weights with our own classifier (dense neural layers in our case), adapted to our problem. We then use the convolutional part of the model as a feature extractor already well trained, allowing us to quickly obtain good results.""")
    st.markdown("""---""")
    col1, col2, col3 = st.columns([1,6,1])    
    with col2:
        st.image(read_image_bucket( image_path + 'TF_black.png'),channels="RGB", output_format="auto")
        st.write("")
        st.write("")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:  
        st.write("""The training of such a model is generally done in the following way, we start by training only the classifier part by freezing the weights of the convolutional part (which are normally almost optimal), this part is called tuning. Once the classifier is well trained, we unfreeze the layers (by packets or all at once) and then we continue training, this part is called fine-tuning. Moreover, we use a lower learning rate during this phase in order to fine-tune the weights that are already near-optimal.""")
    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2: 
        st.title("Preprocessing")
        st.write("")
        st.write("""The pre-trained models we use have been trained on images of a specific format. For the transfer learning to make sense, it was essential to use this same format for our images. Thus, our images are always resized to the size 224x224 (value commonly used for the pre-trained models), the pixel values are rescaled between 0 and 1 and the 3 RGB channels are renormalized to observe the same distribution as the ImageNet images, i.e.""")
        st.write("")
        st.write("")
        st.write("""One of the main drawbacks of Deep Learning models is that they tend to overfit due to the very large number of parameters. One solution to this problem is Data Augmentation. This practice consists in degrading the images of the training set to allow the model to generalize as much as possible. There are many techniques of Data Augmentation, among the most common, we find the rotation, translation, flipping, blurring, etc.. Below, we present an example of Data Augmentation that we used on our images.""")
        st.write("")
        st.write("")
        
    c1,c2,c3 = st.columns([0.5,1,0.5])
    with c2:
        st.image(read_image_bucket( image_path + 'Data_augmentation.png'),channels="RGB", output_format="auto", use_column_width=True)

    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2: 
        st.title("Architecture")
        st.write("")
        st.write("""In the context of transfer learning, the convolutional part of the model is determined by the pre-trained model that we will use. However, we still have to define the architecture of the classifier part. In our case, we have chosen to work with dense neural networks. We then carried out a work of optimization of the architecture which consisted in testing different configurations in order to find the one which gives the best results (note that such a study must be done after the choice of the pre-trained model). The work done is presented in detail in our report (available here). In our project we tested two pre-trained models: VGG19 and DenseNet161. We present in the figure below the architecture chosen for the VGG19 and DenseNet161 models.""")
        st.write("")
        st.write("")
    expander = st.expander("Architecture")
    with st.expander("Architecture") : 
        col1, col2, col3 = st.columns([1,5,1])    
        with col2:
            st.subheader("DenseNet")
            st.image(read_image_bucket( image_path + 'Densenet161.png'),channels="RGB", output_format="auto", use_column_width = 'auto')
            st.write("")
            st.subheader("VGG19")
            st.image(read_image_bucket( image_path + 'VGG19.png'),channels="RGB", output_format="auto", use_column_width = 'auto')
    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2: 
        st.title("Hyperparameters")
        st.write("")
        st.write("""The performance of the models is intrinsically linked to the correct selection of its hyperparameters. Several hyperparameters have been tested and the results are presented in the report. We will come back to the most important hyperparameter in the following: the learning rate.""")
        st.write("")
        st.write("")
    col5, col6, col7, col8 = st.columns([0.5,4,4,0.5])
    with col7:
        st.write("""The choice of an optimal learning rate depends on the topology of the loss function, which in turn depends on the data set and the architecture. To find the optimal learning rate, it is possible to perform several experiments and analyze the results one by one. However, this is very time consuming. Fortunately, in Fastai, there is a learning rate search function called lr_find, which essentially performs a simple experiment where the learning rate is gradually increased after each mini-batch, while recording the loss function at each iteration. The graphical representation of the losses as a function of the learning rate will give us an idea of the evolution of the loss function and can be used as a starting point to find our optimal learning rate. The following figure shows the curve obtained with the lr_find function. """)
        st.write("")
        st.write("")
       
    with col6:
        st.image(read_image_bucket( image_path + 'lr_find.png'),channels="RGB", output_format="auto")
    st.markdown("""---""")

    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2: 
        st.title("Training")
        st.write("")
        st.write("""The architecture and the hyperparameters having been optimized, we could undertake the training of the model. After testing it, we chose to use a particular method of learning rates scheduler (function that allows to change the learning rate during the training) named one cycle fit, available in the Fastai library, and described in the 2018 paper of Smith et al. [https://arxiv.org/pdf/1708.07120.pdf].""")
        st.write("")
        st.write("")
        st.write("""The principle is to increase and decrease the learning rate during training as shown in the figure below.""")
        st.write("")
        st.write("")
    col5, col6, col7, col8 = st.columns([0.5,4,4,0.5])
    with col7:
        st.write("")
        st.write("")
        st.write("")
        st.write("""This method allows the models to converge more quickly and to explore larger learning rates during training (the can be an order of magnitude larger than the optimal learning rate) which is a regularization method reducing overfitting. This method was used for the Densenet161 model (trained on Fastai). For the VGG19 model (trained on Keras) we used a scheduler that decreases the learning rates little by little during the training.""")
        st.write("")
        st.write("")  
    with col6:
        st.image(read_image_bucket( image_path + 'one_cycle_fit.png'),channels="RGB", output_format="auto")
    st.markdown("""---""")
    
    col1, col2, col3 = st.columns([0.5,8,0.5])    
    with col2: 
        st.write("""Finally, we trained our model with the following parameters:""")
        expander = st.expander("Results")
    col1, col2, col3 = st.columns([1,2,1])    
    with col2:
        with st.expander("Results"):
            tab1,tab2= st.tabs(["VGG19", "Densenet161"])
            with tab2: 
                st.image(read_image_bucket( image_path + 'train_densenet.png'),channels="RGB", output_format="auto")
            with tab1: 
                st.image(read_image_bucket( image_path + 'train_vgg19.png'),channels="RGB", output_format="auto")
            
                
    st.write("")
    st.write("")
    st.markdown("""---""")
    
    c1,c2,c3 = st.columns([0.5,8,0.5])
    with c2: 
        st.write("""The figure below shows the evolution of the loss, validation loss and validation accuracy during the training. We observe that the model overfits slightly at the end of the training. The final result obtained is a Top-1 Accuracy of 86.2% and a Top-5 Accuracy of 96.3% on the validation set and a Top-1 Accuracy of 88.2% and a Top-5 Accuracy of 97.2% on a test set made of 2917 images distributed evenly between each class.""")

    expander = st.expander("Results")
    with st.expander("Results"):
        tab1,tab2= st.tabs(["Densenet", "VGG19"])
        with tab1: 
            c1,c2,c3 = st.columns([1,6,1])
            with c2: 
                st.image(read_image_bucket( image_path + 'dense_loss_val_acc_100_2.png'),channels="RGB", output_format="auto", width=1000)
        
        with tab2:
            c1,c2,c3 = st.columns([1,6,1])
            with c2: 
                st.image(read_image_bucket( image_path + 'VGG19_loss_val_acc_100_2.png'),channels="RGB", output_format="auto", width=1000)
    c1,c2,c3 = st.columns([0.5,0.5,0.5])
    with c2: 
        st.image(read_image_bucket( image_path + 'comp_vgg_dense.png'),channels="RGB", output_format="auto", width=500)
    st.markdown("""---""")

if choice == 'Interpretability':
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title('Project report')
        c1, c2, c3, c4 = st.columns(4, gap="large")
    st.markdown("""---""")


    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:   
        st.title('Interpretability of the model')
        st.write("")
        st.write("""We have implemented two tools to try to understand how the model identifies a fungus on an image: the grad-CAM and the Guided Backpropagation. Below we show these two tools in action.""")
        st.write("")
        st.write("")
        st.image(read_image_bucket( image_path + 'top_amanita.png'),channels="RGB", output_format="auto")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2: 
        st.write("")  
        st.write("")  
        st.write("""The grad-CAM is a low-resolution method that is very discriminating in terms of class and allows us to quickly see that the model does not look at anything other than the mushroom. The Guided Backpropagation, on the other hand, is high resolution but does not discriminate between classes (it highlights the important pixels during the prediction). These tools allowed us to validate the model and, in some situations, to understand the bad predictions of the model as we will see in the following.""")
        st.write("")
        st.write("")
    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:   
        st.title('Interpretability of results')
        st.write("")
        st.write("""The above classification report shows quite heterogeneous results, some classes are very well predicted while others are less so.""")
        st.write("")
        st.write("")
        st.write("""The confusion matrix (hardly displayable for 100 classes) allows to find the most confused classes. The table below shows the first 5 largest non-diagonal elements, we give the true label, the prediction and the number of occurrences.""")

    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:   
        st.write("""An exploration of the available images shows that these species are morphologically similar. This explains why the model struggles to distinguish them well. The figure below shows the similarities between these classes.""")
        st.write("")
        st.write("")
    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:   
        st.write("""Another reason that limits the results of the model are poorly labeled images. Either the mushroom in the image does not match the label, or it is very difficult or impossible to distinguish it in the image. Here are some examples.""")
        st.write("")
        st.write("")
        expander = st.expander("Examples")
        with st.expander("Examples"): 
            tab1, tab2, tab3 = st.tabs(["top_loss_1", "top_loss_2", "top_loss_3"])
            with tab1:
                c1,c2,c3 = st.columns([0.5,2,0.5])
                with c2: 
                    st.image(read_image_bucket( image_path + 'top_loss_1.png'),channels="RGB", output_format="auto")
                st.write("")
                st.write("")
                st.subheader("Interpretation")
                st.write("")
                st.write("")
                st.write("""Here we don't really know if there is a mushroom on the image. However, the model makes a prediction with a probability of 77%. The Guided Backpropagtion tool allows to understand the prediction. Indeed, the model seems to focus on leaves, whose shape strongly resembles the predicted class.""")
            
            with tab2:
                c1,c2,c3 = st.columns([0.5,2,0.5])
                with c2:
                    st.image(read_image_bucket( image_path + 'top_loss_2.png'),channels="RGB", output_format="auto")
                st.write("")
                st.write("")
                st.subheader("Interpretation  ")
                st.write("")
                st.write("")
                st.write("""Here the image looks badly labeled. The mushroom does not look like other mushrooms of the same label. However, we observe similarities with the mushrooms of the predicted class.""")
            
            with tab3:
                c1,c2,c3 = st.columns([0.5,2,0.5])
                with c2:
                    st.image(read_image_bucket( image_path + 'top_loss_3.png'),channels="RGB", output_format="auto")
                st.write("")
                st.write("")
                st.subheader("Interpretation  ")
                st.write("")
                st.write("")
                st.write("""Here there are two different mushrooms on the image. The grad-CAM allows us to see that the model makes a prediction on the orange fungus which seems correct. However, the image has been labeled Exidia glandulosa (black mushroom). This explains why this image is predicted incorrectly.""")

