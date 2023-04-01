# Mushroom_recognition
CNN  project for Mushroom recognition (Keras - FastAI - Docker)


# Project's presentation

These project aims to develop an application to provide Mushroom recognition.
These project is based on two axis : 
- The DataScience part :
    - A CNN Keras (VGG19, VGG16, Xception, Densenet)
    - A CNN FastAI (Densenet161)
- The Data Engineering part
    - Docker files for Frontend (Streamlit + API calls)
    - Docker files for backtend (FAST API + CNN prediction)

The current application is deployed on AWS (public and private subnets) according to the following architecture :

## AWS architecture : 

AWS architecture : 

![My Image](Images/AWS.png)

## Micro-service and API :

Objectives : 

![My Image](Images/DE_big.png)

 
Currently deployed : 

![My Image](Images/DE_small.png)


# Models accuracy : 


Densenet - 100 labels :

![My Image](Images/densenet_loss_val_acc_100_2.png)

Densenet - 250 labels :

![My Image](Images/densenet_loss_val_acc_250.png)

VGG19 - 100 labels :

![My Image](Images/VGG19_loss_val_acc_100_2.jpg)



