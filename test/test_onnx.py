import torch
from PIL import Image
import onnx 
import onnxruntime

from src.convert_to_onnx import convert_torchmodel_to_onnx
from pytorch_model import Classifier,BasicBlock
from src.utils import to_numpy

def test_convert_torchmodel_to_onnx():
    mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
    mtailor.load_state_dict(torch.load("./resnet18-f37072fd.pth"))
    mtailor.eval()

    model_onnx=convert_torchmodel_to_onnx(mtailor) 

    ort_session = onnxruntime.InferenceSession("./mtailor.onnx")

    images_dir_to_test=[
        './n01440764_tench.jpeg',
        './n01667114_mud_turtle.JPEG'
    ]
    for img_dir in images_dir_to_test:
        img = Image.open(img_dir)
        inp=mtailor.preprocess_numpy(img).unsqueeze(0) 

        pred_original_mtailor=mtailor.forward(inp)
        argmax_pred=torch.argmax(pred_original_mtailor).item()

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inp)}
        
        ort_outs = ort_session.run(None, ort_inputs)[0][0].argmax()

        assert argmax_pred==ort_outs,'Failed convert model'
    print('Correct convert model ')

if __name__ == '__main__':
    test_convert_torchmodel_to_onnx()




