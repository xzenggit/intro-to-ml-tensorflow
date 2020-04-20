
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
import tensorflow_hub as hub
from PIL import Image


def process_image(image, IMG_SIZE=224):
    """
    Image preprocessing: function should take in an image (in the form of a NumPy array) and 
    return an image in the form of a NumPy array with shape (IMG_SIZE, IMG_SIZE, 3)
    """
    
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    
    return image.numpy()



if __name__=="__main__":    
    
    parser = argparse.ArgumentParser(description='Image classifier app')
    parser.add_argument('image_path', action="store")
    parser.add_argument('saved_model', action="store")
    parser.add_argument('--category_names', action="store", dest="json_map", default='label_map.json')
    parser.add_argument('--top_k', action="store", dest="top_k", type=int, default=5)
    
    results = parser.parse_args()
    json_map = results.json_map
    saved_model = results.saved_model
    image_path = results.image_path
    top_k = results.top_k
    
    # Load model
    model = tf.keras.models.load_model(saved_model, custom_objects={'KerasLayer':hub.KerasLayer})
    # load json name map
    with open(json_map, 'r') as f:
        class_names = json.load(f)
    
    # Load the image
    im = Image.open(image_path)
    test_image = np.asarray(im)
    
    processed_test_image = process_image(test_image)
    p = model.predict(np.expand_dims(processed_test_image, axis=0))
    
    p = tf.nn.softmax(p[0])
    classes = np.argsort(p)[-top_k:]
    
    print(p.numpy()[classes])
    strclasses = [str(x+1) for x in classes]
    print(strclasses)
    print([class_names[x] for x in strclasses])