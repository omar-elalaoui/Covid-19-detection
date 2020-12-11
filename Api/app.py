import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask, render_template
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)


def get_model():
    global model_inV3, model_resNet50, model_VGG16, model_DenseNet201
    model_inV3 = load_model('./InceptionV3.h5')
    model_resNet50 = load_model('./ResNet50.h5')
    model_VGG16 = load_model('./VGG16.h5')
    model_DenseNet201 = load_model('./DenseNet201.h5')
    print("models loaded!!!!")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image= image.resize(target_size)
    image = np.array(image) / 255.
    image = (np.expand_dims(image, 0))
    return image

print(" loading keras model ....")
get_model()



@app.route("/hello", methods=["GET"])
def hello():
    return "hello word"


@app.route("/predict", methods=["POST"])
def predict():
    #Save codes and corresponding categories in a dictionary
    inverse_dict = {0: 'Covid', 1: 'Normal'}
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(512, 512))
    #Converting predictions to probability values
    
    
    result_inV3 = np.squeeze(model_inV3.predict(processed_image))
    predict_class_inV3 = np.argmax(result_inV3)
    
    result_resNet50 = np.squeeze(model_resNet50.predict(processed_image))
    predict_class_resNet50 = np.argmax(result_resNet50)
    
    result_VGG16 = np.squeeze(model_VGG16.predict(processed_image))
    predict_class_VGG16 = np.argmax(result_VGG16)
    
    result_densNet = np.squeeze(model_DenseNet201.predict(processed_image))
    predict_class_densNet = np.argmax(result_densNet)
    
    
    response = {
        'inceptionV3': {
            'class': str(inverse_dict[int(predict_class_inV3)]),
            'value': str(result_inV3[predict_class_inV3])
        },
        
        'resNet50': {
            'class': str(inverse_dict[int(predict_class_resNet50)]),
            'value': str(result_resNet50[predict_class_resNet50])
        },
        
        'VGG16': {
            'class': str(inverse_dict[int(predict_class_VGG16)]),
            'value': str(result_VGG16[predict_class_VGG16])
        },
        
        'danseNet201': {
            'class': str(inverse_dict[int(predict_class_densNet)]),
            'value': str(result_densNet[predict_class_densNet])
        }
    }
    return jsonify(response)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080")


