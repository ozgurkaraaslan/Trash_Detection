# Trash_Detection

This project aims to classify different types of waste products. These types are: Glass, Plastic, Metal, Trash, Cardboard, and Paper.

## Usage of the Trained Model

- Download the repository.

- Go to Trained_Model folder within the terminal.

- Run the main.py file with `python ./main.py [IMAGE_PATH]` or `python ./main.py [IMAGE_PATH]`

> Example of [IMAGE_PATH]: `test_images/unknown_1.jpg`. You can give any desired image.

## Training the Model from Scratch

- Download data to the same path with `model.py`

- Run the model.py file with `python ./model.py` or `python3 ./model.py`

- To use your trained model, change `class_indices_json` and `trash_classifier.h5` with your created files.
