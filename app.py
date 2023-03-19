from typing import Union

from PIL import Image
import torch

from src.model import OnnxModel

def init():
    model=OnnxModel()
   

def inference(
    path:Union[str,Image.Image]
)->dict:
    global model 
    model=OnnxModel()   
    return {'res':model(path)}
