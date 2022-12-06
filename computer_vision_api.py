import fastapi
import uvicorn
import torch
import base64
import json

from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

class ImageClassifier:

    def __init__(self, image=None):
        if image == None:
            self.input_image = Image.open('current_image.jpg')
        else:
            self.input_image = Image.open(image)

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.model.eval()
        self.preprocessing()

    def preprocessing(self):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(self.input_image)
        self.input_batch = input_tensor.unsqueeze(0)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            self.input_batch = self.input_batch.to('cuda')
            self.model.to('cuda')

    def predict(self):
        with torch.no_grad():
            output = self.model(self.input_batch)   # Tensor of shape 1000 with confidence scores for Imagenet's classes

        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]

        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        prob_dictionary = {}

        for i in range(top5_prob.size(0)):
            prob_dictionary[categories[top5_catid[i]]] = top5_prob[i].item()

        return prob_dictionary

class ImageData64(BaseModel):
    encoded_string: str

api = fastapi.FastAPI()

@api.post("/predict/image64")
def predict_image64(image: ImageData64):
    try:
        imgdata = base64.b64decode(image.encoded_string)
        filename = 'current_image.jpg'
        with open(filename, 'wb') as f:
            f.write(imgdata)

        image_classifier = ImageClassifier(filename)

        return image_classifier.predict()
    except:
        return_dict = {
            'Status': 'Error',
            'Fix': 'Ensure that the image has been base64 encoded and then decoded into a string which is sent as a parameter of requests.post --> data=json.dumps({"encoded_string": encoded_string})'
        }
        return fastapi.Response(content=json.dumps(return_dict),
                                media_type='application/json',
                                status_code=400)

@api.post("/predict/image")
def predict_image(image: fastapi.UploadFile = fastapi.File(...)):
    try:
        image_classifier = ImageClassifier(image.file)

        return image_classifier.predict()
    except:
        return_dict = {
            'Status': 'Error',
            'Fix': 'Ensure that the image file is added as a parameter to requests.post --> files={"image": open("filename.ext", "rb")}'
        }
        return fastapi.Response(content=json.dumps(return_dict),
                        media_type='application/json',
                        status_code=400)

if __name__ == "__main__":  
    uvicorn.run(api)