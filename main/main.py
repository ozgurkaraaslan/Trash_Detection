import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.models import load_model
import json

# Load the trained model
model = load_model("trash_classifier.h5")

# Load class indices
with open("class_indices.json") as f:
    class_indices = json.load(f)


# Define the function to preprocess the image
def prepare_image(file_path):
    img = tf_image.load_img(file_path, target_size=(180, 180), color_mode="rgb")
    img_array = tf_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Now use class_indices in your prediction function
def predict_image_class(file_path):
    prepared_img = prepare_image(file_path)
    result = model.predict(prepared_img)
    predicted_class = None
    for key, value in class_indices.items():
        if value == np.argmax(result):
            predicted_class = key
            break
    return predicted_class


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        prediction = predict_image_class(image_path)
        print(f"\nPredicted Class for '{image_path}':", prediction)
    else:
        print("Please provide an image path.")
