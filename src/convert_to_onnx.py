# Some standard imports
import io
import sys

import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import onnx

def convert_torchmodel_to_onnx(
    torchmodel:nn.Module,
):
    torch.onnx.export(
        model = torchmodel,
        args=torch.rand((1, 3, 224, 224)),
        f = './mtailor.onnx',
        export_params= True,
        input_names = ['input'],
        output_names = ['output']
    )

    onnx_model = onnx.load("./mtailor.onnx")
    
    return onnx_model



