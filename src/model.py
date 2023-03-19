from typing import Any,Union

import PIL
from PIL import Image
import torch
import numpy as np
import onnx 
import onnxruntime

from src.convert_to_onnx import convert_torchmodel_to_onnx
from pytorch_model import Classifier,BasicBlock
from src.utils import to_numpy

NUM_LINE_LABEL_FILE=1000

def preprocessing_image(
    img:Union[str,Image.Image]
)->np.ndarray:
    if isinstance(img,str):
        img = Image.open(img)
    elif isinstance(img,Image.Image):
        pass

    inp=Classifier.preprocess_numpy(img).unsqueeze(0) 
    return inp


class OnnxModel():
    def __init__(self,model_dir:str='./mtailor.onnx') -> None:
        self.ort_session=onnxruntime.InferenceSession(model_dir)
        self.mapping_dict=self._load_label_imagenet()

    def _load_label_imagenet(self,file_label_dir:str='./imagenet1000_clsidx_to_labels.txt'):
        """Load label of imagenet dataset

        Args:
            file_label_dir (str, optional): direction to label description file. Defaults to './imagenet1000_clsidx_to_labels.txt'.

        
        """
        d={}
        with open(file_label_dir) as f:
            f.readline()
            for _ in range(NUM_LINE_LABEL_FILE):
                (key, val) = f.readline().split(':')
                d[int(key)] = val
        return d

    def __call__(self,img:Union[str,Image.Image]) -> Any:
        inp=preprocessing_image(img)

        ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(inp)}
        
        ort_outs = self.ort_session.run(None, ort_inputs)[0][0].argmax()
        return self.mapping_dict[ort_outs]
    
# preprocessing_image('./n01440764_tench.jpeg')

# model=OnnxModel()
# print(model('./n01667114_mud_turtle.JPEG'))
