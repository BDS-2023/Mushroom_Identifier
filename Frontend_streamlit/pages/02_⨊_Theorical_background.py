import streamlit as st
import base64
from PIL import Image
import pandas as pd
import numpy as np
import s3fs
import plotly.express as px
import io

st.set_page_config(
    page_title="Hello Mushroom-dawan",
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




choice = st.sidebar.radio("Submenu", ["Introduction","Données", "Montage des machines virtuelles", "Classification", "Interprétabilité"])
if choice == 'Introduction':
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title('Rapport de projet')
        c1, c2, c3, c4 = st.columns(4, gap="large")
    st.markdown("""---""")


    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:   
        st.title('Contexte')
        st.markdown("---")
        st.write("""Il existe différentes techniques permettant d’identifier des espèces de champignons. La plus utilisée et la plus ancienne est l'identification morphologique, qui classifie les individus par leurs caractéristiques anatomiques. Cependant, cette technique a l'inconvénient d’être limitée en termes de précision car dépendante de l'observation et du protocole d'identification de la personne qui la pratique.""")
        st.write("")
        st.write("")
        st.write("")
        st.write("""L’objectif du projet « Reconnaissance de champignons » était de faire de la reconnaissance automatisée de champignons au travers des technologies de Computer Vision. Il s’inscrivait dans le cadre de la formation Data Scientist dispensée par la société DataScientest et constituait pour nous une première expérience professionnelle en vue de valoriser nos connaissances et de monter en compétence dans le domaine de la Data Science. Plus précisément, ce projet s’inscrivait dans le domaine de la computer vision et avait pour objectif d’acquérir des connaissances sur les technologies de Deep Learning.""")
        
if choice == 'Données':
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title('Rapport de projet')
        c1, c2, c3, c4 = st.columns(4, gap="large")
    st.markdown("""---""")



    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:   
        st.title('Données')
        st.write("")
        st.write("""Pour le projet nous avons travaillé avec des images téléchargées depuis les bases de données disponibles sur le site https://www.gbif.org/. GBIF est un projet scientifique visant à répertorier l’ensemble des données taxonomiques sur la biodiversité, dont le Royaume Fungi. Le portail du GBIF expose une API permettant de télécharger des photos d’espèces vivantes à partir d’une clef taxonomique. Il existe, sur le site GBIF, un fichier qui permet de lier les clefs taxonomiques aux noms des espèces ainsi qu’à leurs taxonomies complètes. Voici quelques lignes du fichier GBIF_tax.csv pour les entrées appartenant au Kingdom Fungi.""")
        st.write("")
        st.write("")
        st.write("")
        expander = st.expander("Mushrooms DataFrame")
        
        expander.dataframe(df)
        expander.write("""La colonne numberOfOccurences indique le nombre d’images disponibles pour chaque espèce. Au total la source expose environ 34 M de photos de champignons.""")
    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2: 
        st.title("Exploration de données")
        st.write("")
        st.write("""La première étape était de définir notre jeu de données. Il était nécessaire de définir les classes à utiliser pour entrainer un modèle. Une photo représente un champignon qui est définit par son nom ainsi que par sa taxonomie. En biologie, une taxonomie est un moyen de définir des groupes hiérarchiques basés sur des caractéristiques communes (voir figure).  Plus on descend dans un arbre taxonomique plus l’identification devient spécifique.  """)
        st.write("")
        st.write("")
        st.write("")
    col5, col6, col7, col8 = st.columns([0.5,4,4,0.5])
    with col6 : 
        st.image(read_image_bucket( image_path + 'tax_black.png'),channels="RGB", output_format="auto")
    st.markdown("""---""")
    with col7: 
        st.write("""Dans le tableau nous trouvons la colonne taxonRank qui indique le dernier niveau atteint dans l’identification des photos des champignons pour une certaine taxonKey. A chaque rang taxonomique (ex : Family) nous trouvons un certain nombre de classe, nombre qui augmente exponentiellement lorsque nous descendons dans l’arbre. La figure ci-dessous est un sunburst représentant l’ensemble des rangs taxonomique (en partant du Kingdom Fungi) ainsi que l’ensemble des classes dans chaque rang.""")
    st.write("")
    st.write("")

    df4 = df.dropna(subset = ['kingdom', 'phylum', 'class','order','family','genus'])
    fig2 = px.sunburst(df4, path=['kingdom','phylum','class','order','family','genus'],values=df4.numberOfOccurrences,maxdepth=3,color_discrete_sequence=px.colors.qualitative.T10)
    c1,c2,c3 = st.columns([0.2,4,0.2])
    with c2: 
        st.plotly_chart(fig2 ,width= 900) 
        
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:
        

        st.write("La figure ci-dessous, quant à elle, représente le nombre de classes disponibles dans chaque rang taxonomique.")
              
    c1,c2,c3 = st.columns([0.5,1,0.5])
    with c2:
        df_forbar = df2.groupby("rank").count().sort_values(by='occurence')
        df_forbar['rank'] = df_forbar.index.values
        fig3 = px.bar(data_frame=df_forbar,x='rank',y='occurence',barmode='group',log_y=True,hover_data=["occurence"],
             labels={'rank':'Rang taxonomique','occurence':'Nombre de taxons'},title="Nombre de taxons à chaque rang taxonomique")
        st.plotly_chart(fig3)

        st.write("")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:
        st.write("""Ces figures permettent de mettre en avant une des principales problématiques du projet : le nombre faramineux d’espèces de champignons différentes. En effet, sur GBIF, nous trouvons 117 539 espèces de champignon. Une étude de l’état de l’art sur la classification par Computer Vision montre qu’un modèle capable d’identifier autant de classes est aujourd’hui hors de portée technologiquement parlant.""")
    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:  
        st.write("")
        st.write("")
        st.write("""Afin de réduire le nombre de classe disponible, il peut paraître judicieux de se placer haut dans l’arbre taxonomique. Par exemple, il existe uniquement 7 Phylums différents pour les champignons. Cependant, en travaillant haut dans l’arbre taxonomique, nous nous retrouvons avec des classes très générales, dans lesquelles les champignons peuvent être très différents. Un exemple est donné dans la figure ci-dessous sur laquelle nous présentons deux champignons appartenant à la même Family : Agaricaceae.  """)
    st.write("")
    st.write("")
    col5, col6, col7 = st.columns([0.5,8,0.5])
    with col6 : 
        st.image(read_image_bucket( image_path + 'Agaricaceae.png'),channels="RGB", output_format="auto")
    
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:
        st.write("""Nous avons estimé que travailler avec de telles données pourrait rendre l’apprentissage très difficile, car nous demandons au modèle de regrouper dans la même classe des images très différentes.  """)
        st.write("")
        st.write("")
        st.write("""Finalement, nous avons décidé de travailler à la maille Species afin d’avoir, dans une même classe, des champignons qui se ressemblent, même si cela entraine une augmentation du nombre de classes disponibles. Pour le nombre de classes, nous nous sommes fixés comme objectif de travailler avec les 100 classes les plus représentées dans GBIF. La figure ci-dessous montre le nombre d’images disponibles pour chacune de ces classes.
""")
    st.write("")
    st.write("")
    
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:
        st.write("""Ici, nous voyons que le nombre d’images disponible ne posera pas de problème. Il est même nécessaire de se restreindre afin de pouvoir entrainer des modèles sur nos machines dans des temps acceptables. Nous avons donc choisi de travailler avec environ 500 images par espèces, soit un total de 50000 images.""")
    
    dft = df2 
    dft_temp = dft[dft['rank'] == 'Species'].head(100)
    fig4 = px.bar(data_frame=dft_temp,x='name',y='occurence',barmode='group',log_y=False,hover_data=["occurence"],color="occurence",
             labels={'name':'Taxon','occurence':"Nombre d'images"},title="Nombre d'images disponibles pour les 100 premiers taxon du rang Species", height=800, width=1400)
    st.plotly_chart(fig4)


if choice == 'Montage des machines virtuelles':
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title('Rapport de projet')
    st.markdown("""---""")
    expander3 = st.expander("Dans le cadre de notre projet voici le modèle de machine virtuelle que nous avons souhaité développer :") 
    with st.expander("Dans le cadre de notre projet voici le modèle de machine virtuelle que nous avons souhaité développer :") :
        col1, col2, col3 = st.columns([0.5,2,0.5])
        with col2:
            st.image(read_image_bucket( image_path + 'DE_big.png'),channels="RGB", output_format="auto", use_column_width = 'auto')
            """
* Frontend : 
  * Streamlit permettant la réception d'image classifier
    * Host = EC2 Public
* Backend : 
  * Classification des photos reçues par le Streamlit
    * Host = EC2 Privé
        * Télechargement des images de la base MO
        * Télechargement des images de la base GBIF
        * Mise à jour des données reçues sur la base MO
        * Calculs des modèles de prédiction NVIDIA
"""

    st.markdown("""---""")

    expander3 = st.expander("Pour optimiser notre temps sur les modèles, nous avons finalement effectué cette architecture :") 
    with st.expander("Pour optimiser notre temps sur les modèles, nous avons finalement effectué cette architecture :") :
        col1, col2, col3 = st.columns([0.5,2,0.5])
        with col2:
            st.image(read_image_bucket( image_path + 'DE_small.png'),channels="RGB", output_format="auto", use_column_width = 'auto')
            """
* Frontend : 
  * Streamlit permettant la réception d'image classifier
    * Host = EC2 Public
* Backend : 
  * Classification des photos reçues par le Streamlit
    * Host = EC2 Privé
        * Télechargement des images de la base GBIF
        * Mise à jour des données reçues sur la base MO
"""


if choice == 'Classification':
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title('Rapport de projet')
        c1, c2, c3, c4 = st.columns(4, gap="large")
    st.markdown("""---""")



    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:   
        st.title('Méthodes')
        st.write("")
        st.write("""Comme mentionné précédemment, ce projet vise à identifier des images de champignons. Ce procédé est un problème typique de classification d'images, généralement traité par des méthodes de Deep Learning et plus spécifiquement à l’aide de réseaux de neurones convolutifs (CNN). De plus, il est aujourd’hui d’usage d’utiliser les méthodes d’apprentissage par transfert (transfer learning) afin de réduire considérablement le temp d’entrainement des modèles.""")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("""Le principe est simple, nous utilisons le savoir acquis par des modèles entrainés sur des machines très puissantes et pendant des temps longs pour une tâche particulière afin de résoudre un problème différent mais qui présente des similitudes. Pour les CNN, il existe aujourd’hui de nombreux modèles (VGG, MobileNet, ResNet, DenseNet, etc.) qui ont été entraînés sur le jeu de données ImageNet, composé de 1000 classes et de 1 281 167 images. Le principe de l’apprentissage par transfert (représenté sur la figure ci-dessous) est d’interfacer la partie convolutive des modèles dit pré-entraînés ainsi que leurs poids avec notre propre classifieur (couches de neurones denses dans notre cas), adapté à notre problème. Nous utilisons alors la partie convolutive du modèle comme extracteur de features déjà très bien entraîné, permettant ainsi d’obtenir rapidement de bons résultats.
""")
    st.markdown("""---""")
    col1, col2, col3 = st.columns([1,6,1])    
    with col2:
        st.image(read_image_bucket( image_path + 'TF_black.png'),channels="RGB", output_format="auto")
        st.write("")
        st.write("")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:  
        st.write("""L’entrainement d’un tel modèle se fait généralement de la manière suivante, on commence par entraîner uniquement la partie classificatrice en gelant (freeze) les poids de la partie convolutive (qui sont normalement quasi-optimaux), cette partie est nommée tuning. Une fois le classificateur bien entraîné, nous dégelons (unfreeze) les couches (par paquets ou toutes d’un coup) puis nous continuons l’entraînement, cette partie est nommée fine-tuning. De plus, nous utilisons un learning rate plus faible lors de cette phase afin d’ajuster plus finement les poids déjà quasi-optimaux.""")
    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2: 
        st.title("Preprocessing")
        st.write("")
        st.write("""Les modèles pré-entrainés que l’on utilise ont été entrainés sur des images d’un format spécifique. Pour que le transfer learning ait du sens, il était indispensable d’utiliser ce même format pour nos images. Ainsi, nos images sont toujours redimensionnées à la taille 224x224 (valeur couramment utilisée pour les modèles pré-entrainés), la valeur des pixels est rééchelonnées entre 0 et 1 puis les 3 canaux RGB sont renormalisés pour observer la même distribution que les images de ImageNet, i.e.  et .""")
        st.write("")
        st.write("")
        st.write("""Un des principaux défauts des modèles de Deep Learning est qu’ils ont tendance à overfit en raison du très grand nombre de paramètres. Une des solutions pour pallier ce problème est le Data Augmentation. Cette pratique consiste à dégrader les images du jeu d’entrainement afin de permettre au modèle de généraliser le plus possible. Il existe de nombreuses techniques de Data Augmentation, parmi les plus courantes, nous retrouvons la rotation, la translation, le retournement, le floutage, etc. Ci-dessous, nous présentons un exemple de Data Augmentation que nous avons utilisés sur nos images. """)
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
        st.write("""Dans le cadre du transfer learning, la partie convolutive du modèle est déterminée par le modèle pré-entrainé que nous allons utiliser. Cependant, il reste à définir l’architecture de la partie Classifieur. Dans notre cas, nous avons choisi de travailler avec des réseaux de neurones denses. Nous avons ensuite réalisé un travail d’optimisation de l’architecture qui a consisté à tester différentes configurations afin de trouver celle qui donne les meilleurs résultats (notons qu’une telle étude doit être faite après le choix du modèle pré-entrainé). Les travaux effectués sont présentés en détail dans notre rapport (disponible ici). Dans le cadre de notre projet nous avons testé deux modèles pré-entrainés : VGG19 et DenseNet161. Nous présentons dans la figure ci-dessous l’architecture retenue pour le modèle VGG19 et DenseNet161.""")
        st.write("")
        st.write("")
    expander = st.expander("Architecture")
    with st.expander("Architecture") : 
        col1, col2, col3 = st.columns([1,5,1])    
        with col2:
            st.subheader("DenseNet")
            st.image(read_image_bucket( image_path + 'Densenet161.png'),channels="RGB", output_format="auto", width=1000)
            st.write("")
            st.subheader("VGG19")
            st.image(read_image_bucket( image_path + 'VGG19.png'),channels="RGB", output_format="auto", width=1000)
    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2: 
        st.title("Hyperparamètres")
        st.write("")
        st.write("""Les performances des modèles sont intrinsèquement liées à la bonne sélection de ses hyperparamètres. Plusieurs hyperparamètres ont été testés et les résultats sont présentés dans le rapport. Nous allons revenir dans la suite sur l’hyperparamètre le plus important : le learning rate.""")
        st.write("")
        st.write("")
    col5, col6, col7, col8 = st.columns([0.5,4,4,0.5])
    with col7:
        st.write("""Le choix d'un learning rate optimal dépend de la topologie de la fonction de perte, qui est elle-même fonction de l'ensemble des données et de l'architecture. Pour trouver le learning rate optimal, il est possible de réaliser plusieurs expériences et d'analyser les résultats un par un. Cependant, cela prend beaucoup de temps. Heureusement, dans Fastai, il existe une fonction de recherche de learning rate appelée lr_find, qui effectue essentiellement une expérience simple où le learning rate est progressivement augmenté après chaque mini batch, tout en enregistrant la fonction de perte à chaque itération. La représentation graphique des pertes en fonction du taux d'apprentissage nous donnera une idée de l'évolution de la fonction de perte et pourra être utilisée comme point de départ pour trouver notre taux d'apprentissage optimal. La figure suivante montre la courbe obtenu grâce à la fonction lr_find. 
 """)
        st.write("")
        st.write("")
       
    with col6:
        st.image(read_image_bucket( image_path + 'lr_find.png'),channels="RGB", output_format="auto")
    st.markdown("""---""")

    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2: 
        st.title("Entrainement")
        st.write("")
        st.write("""L’architecture et les hyperparamètres ayant été optimisé, nous pouvions entreprendre l’entrainement du modèle. Après l’avoir testé, nous avons choisis d’utiliser une méthode particulière de learning rates scheduler (fonction qui permet de changer le learning rate au cours de l’entrainement) nommé one cycle fit, disponible dans la bibliothèque Fastai, et décrite dans l’article de 2018 de Smith et al. [https://arxiv.org/pdf/1708.07120.pdf].""")
        st.write("")
        st.write("")
        st.write("""Le principe est d’augmenter puis de diminuer le learning rate au cours de l’entrainement comme représenté sur la figure ci-dessous.""")
        st.write("")
        st.write("")
    col5, col6, col7, col8 = st.columns([0.5,4,4,0.5])
    with col7:
        st.write("")
        st.write("")
        st.write("")
        st.write("""Cette méthode permet de faire converger les modèles plus rapidement et permet d’explorer des learning rate plus grands au cours de l’entrainement (le  peut être un ordre de grandeur plus grand que le learning rate optimal) ce qui constitue une méthode de régularisation réduisant l’overfitting. Cette méthode a été utilisée pour le modèle Densenet161 (entrainé sur Fastai). Pour le modèle VGG19 (entrainé sur Keras) nous avons utilisé un scheduler qui diminue le learning rates petit à petit au cours de l’entrainement. """)
        st.write("")
        st.write("")  
    with col6:
        st.image(read_image_bucket( image_path + 'one_cycle_fit.png'),channels="RGB", output_format="auto")
    st.markdown("""---""")
    
    col1, col2, col3 = st.columns([0.5,8,0.5])    
    with col2: 
        st.write("""Finalement, nous avons entrainés notre modèle avec les paramètres suivants :""")
        expander = st.expander("Résultats")
    col1, col2, col3 = st.columns([1,2,1])    
    with col2:
        with st.expander("Résultats"):
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
        st.write("""La figure ci-dessous montre l’évolution des loss, validation loss et validation accuracy au cours de l’entrainement. Nous observons que le modèle overfit légèrement en fin d’entrainement. Le résultat final obtenu est une Top-1 Accuracy de 86.2% et une Top-5 Accuracy de 96.3% sur le jeu de validation et Top-1 Accuracy de 88.2% et une Top-5 Accuracy de 97.2% sur un jeu de test constitué de 2917 images réparties équilibrèrent entre chaque classe.""")

    expander = st.expander("Résultats")
    with st.expander("Résultats"):
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

if choice == 'Interprétabilité':
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title('Rapport de projet')
        c1, c2, c3, c4 = st.columns(4, gap="large")
    st.markdown("""---""")


    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:   
        st.title('Interprétabilité du modèle')
        st.write("")
        st.write("""Nous avons mis en place deux outils afin de tenter de comprendre comment le modèle identifie un champignon sur une image : le grad-CAM et le Guided Backpropagation. Ci-dessous nous montrons ces deux outils en action.""")
        st.write("")
        st.write("")
        st.image(read_image_bucket( image_path + 'top_amanita.png'),channels="RGB", output_format="auto")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2: 
        st.write("")  
        st.write("")  
        st.write("""Le grad-CAM est une méthode basse résolution très discriminante en termes de classe qui permet rapidement de voir que le modèle ne regarde pas autre chose que le champignon. Le Guided Backpropagation, quant à lui, est à haute résolution mais ne permet pas de discriminer les classes (il met en valeur les pixels important lors de la prédiction). Ces outils nous on permis de valider le modèle et, dans certaines situations, de comprendre les mauvaises prédictions du modèle comme nous allons le voir dans la suite.""")
        st.write("")
        st.write("")
    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:   
        st.title('Interprétabilité des résultats')
        st.write("")
        st.write("""Le rapport de classification ci-dessus montre des résultats assez hétérogènes, certaines classes sont très bien prédites alors que d’autres le sont moins""")
        st.write("")
        st.write("")
        st.write("""La matrice de confusion (difficilement affichable pour 100 classes) permet de trouver les classes les plus confuses. Le tableau ci-dessous montre les 5 premiers éléments non diagonaux les plus grands, nous donnons le vrai label, la prédiction ainsi que le nombre d’occurrences.""")

    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:   
        st.write("""Une exploration des images disponibles montre que ces espèces se ressemblent morphologiquement. Ce qui explique pourquoi le modèle peine à bien les distinguer. La figure ci-dessous montre les similitudes entre ces classes.""")
        st.write("")
        st.write("")
    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.5,8,0.5])
    with col2:   
        st.write("""Une autre raison qui limite les résultats du modèle sont les images mal labelisées. Soit le champignon sur l’image ne correspond pas au label, soit il est très difficile voire impossible de le distinguer sur l’image. Voici quelques exemples.""")
        st.write("")
        st.write("")
        expander = st.expander("Exemples")
        with st.expander("Exemples"): 
            tab1, tab2, tab3 = st.tabs(["top_loss_1", "top_loss_2", "top_loss_3"])
            with tab1:
                c1,c2,c3 = st.columns([0.5,2,0.5])
                with c2: 
                    st.image(read_image_bucket( image_path + 'top_loss_1.png'),channels="RGB", output_format="auto")
                st.write("")
                st.write("")
                st.subheader("Interprétation ")
                st.write("")
                st.write("")
                st.write("""Ici on ne sait pas vraiment s’il y a un champignon sur l’image. Le modèle fait tout de même une prédiction avec une probabilité de 77%. L’outil Guided Backpropagtion permet de comprendre la prédiction. En effet, le modèle semble se concentrer sur des feuilles, dont la forme ressemble fortement à la classe prédite.""")
            
            with tab2:
                c1,c2,c3 = st.columns([0.5,2,0.5])
                with c2:
                    st.image(read_image_bucket( image_path + 'top_loss_2.png'),channels="RGB", output_format="auto")
                st.write("")
                st.write("")
                st.subheader("Interprétation  ")
                st.write("")
                st.write("")
                st.write("""Ici l’image semble mal labélisée. Le champignon ne ressemble pas aux autres champignons de même label. Cependant, nous observons des similitudes avec les champignons de la classe prédite.""")
            
            with tab3:
                c1,c2,c3 = st.columns([0.5,2,0.5])
                with c2:
                    st.image(read_image_bucket( image_path + 'top_loss_3.png'),channels="RGB", output_format="auto")
                st.write("")
                st.write("")
                st.subheader("Interprétation  ")
                st.write("")
                st.write("")
                st.write("""Ici il y a deux champignons différents sur l’image. Le grad-CAM permet de de voir que le modèle fait une prédiction sur le champignon orange qui semble correct. Cependant, l’image a été  labelisée Exidia glandulosa (champignon noir). Ce qui explique pourquoi cette image est mal prédite.""")

