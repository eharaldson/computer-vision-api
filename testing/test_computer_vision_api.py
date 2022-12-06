import requests
import base64
import json

# POST file using base64 encoding
def post_using_base64():
    with open("dog.jpg", "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
        encoded_string = encoded_bytes.decode('utf-8')

    response = requests.post("http://127.0.0.1:8000/predict/image64", data=json.dumps({'encoded_string': encoded_string}))
    print(response.json())

# POST file using fastapi UloadFile class
def post_using_file():
    files = {'image': open("dog.jpg", "rb")}
    response = requests.post("http://127.0.0.1:8000/predict/image", files=files)
    print(response.json())

if __name__ == "__main__":
    post_using_file()