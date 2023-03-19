# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

ENV PYTHONPATH /


# Install git
RUN apt-get update && apt-get install -y git

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ADD resnet18-f37072fd.pth .
ADD n01440764_tench.jpeg .
ADD n01667114_mud_turtle.JPEG .
ADD imagenet1000_clsidx_to_labels.txt .



# Copy src, contains code 
COPY src/ src/
ADD pytorch_model.py .

COPY test/ test/

RUN python3 test/test_onnx.py


# We add the banana boilerplate here
ADD server.py .

ADD download.py .

RUN python3 download.py


# Add your custom app code, init() and inference()
ADD app.py .

EXPOSE 8000

CMD python3 -u server.py
