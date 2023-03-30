# backend/main.py


import uvicorn
from fastapi import File
from fastapi import FastAPI, Header, Response
from fastapi import UploadFile
import numpy as np
from PIL import Image
import cv2
import base64
from fastapi.exceptions import HTTPException
import io
from fastai.vision.all import *
import pathlib
import boto3
from module import return_cam

#OS adaptation (model developped on Window, but EC2 runnng on Linus)
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

session = boto3.Session()

#Opening a Boto session to get acces to the bucket S3 of the projet
#The EC2 machine has to be configured with AWS CLI and access Key
s3 = session.resource('s3')
my_bucket = s3.Bucket('imagemobucket')

#Model and label dictionnary loading 
model = io.BytesIO(my_bucket.Object("models/mushroom_densenet161_100cat_top_no_data_augmentation.pkl").get()['Body'].read())
learn = load_learner(model, cpu=True)
dict_label = {k: learn.dls.vocab.o2i[k] for k in list(learn.dls.vocab.o2i)}
inv_dict_label = {v: k for k, v in dict_label.items()} 
    

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

""""
API function: The goal is to retrieve from the streamlit API call the image data and to return the classification of the picture
The prediction function takes as input an image and return a prediction array and a GradCam picture.

"""

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile = File (...)):

    if not file:
        return {"message": "No upload file sent"}
    else:
        # if file.content_type != "image/jpeg":#'multipart/form-data':
        #     raise HTTPException(400, detail= 'Invalide image type2')
        
        image = np.array(Image.open(file.file))

        preds = learn.predict(image)[2]
        
        idx_top = torch.topk(preds, 5).indices.detach().numpy()
        name_top = [inv_dict_label[id] for id in idx_top]
        proba_top = torch.topk(preds, 5).values.detach().numpy()
        
        predsstr = ''
        for name,prb in zip(name_top,proba_top):
            predsstr += "{0}:{1:.2f},".format(name,prb*100)

        grad = return_cam(learn,image)
        map_img = np.uint8(grad*255)
        heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR)

        cv_im = cv2.cvtColor(np.uint8(image[:,:,::-1]), cv2.COLOR_RGB2BGR)
        heatmap_img_resi = cv2.resize(heatmap_img,(cv_im.shape[1],cv_im.shape[0]))
        added_image = cv2.addWeighted(cv_im,1,heatmap_img_resi,0.4,0)

        grad = Image.fromarray(np.uint8(added_image))
        b = io.BytesIO()
        grad.save(b, 'jpeg')
        im_bytes = b.getvalue()
        encoded_image = base64.b64encode(im_bytes)

        
    return {'content': predsstr[:-1],'grad-cam': encoded_image, 'test':'Hello World'}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)