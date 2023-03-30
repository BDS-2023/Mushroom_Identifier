from fastai.vision.all import *
import numpy as np

"""
This module.py allows the GradCam calculation and is used during a POST/uploadfile
"""
def return_cam(learn,image):
        x, = first(learn.dls.test_dl([image]))
        m = learn.model.eval()
        # output = m(x)
        
        def hooked_backward():
                with hook_output(m[0][-1][-1]) as hook_a: 
                        with hook_output(m[0][-1][-1], grad=True) as hook_g:
                                preds = m(x)
                                cat = preds.argmax()
                                preds[0,int(cat)].backward()
                return hook_a,hook_g
        
        hook_a,hook_g = hooked_backward()

        acts  = hook_a.stored[0].cpu() #activation maps
        acts.shape

        grad = hook_g.stored[0][0].cpu() #gradients
        # grad.shape

        grad_chan = grad.mean(1).mean(1) # importance weights
        # grad_chan.shape

        mult = F.relu(((acts*grad_chan[...,None,None])).sum(0)) # GradCAM map

        def minmax_norm(x):
                return (x - np.min(x))/(np.max(x) - np.min(x))

        def scaleup(x,size):
                scale_mult=size/x.shape[0]
                upsampled = scipy.ndimage.zoom(x, scale_mult)
                return upsampled

        hmap = mult.detach().cpu()

        hmap_scaleup = minmax_norm(scaleup(hmap,224))
        
        return hmap_scaleup