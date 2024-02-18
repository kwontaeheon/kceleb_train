import os

import numpy as np
import tensorflow as tf

assert tf.__version__.startswith("2")

import matplotlib.pyplot as plt
from tflite_model_maker import image_classifier, model_spec
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

epoch = 300
# spec = model_spec.get('efficientnet_lite4')
spec = image_classifier.ModelSpec(
    uri="https://www.kaggle.com/models/google/efficientnet-v2/frameworks/TensorFlow2/variations/imagenet1k-b3-feature-vector/versions/2"
)
spec.input_image_shape = [300, 300]
bsize = 20


image_path = "/home/terry/code/celebme/celebme_model_202401/faces/women/"
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

# model = image_classifier.create(train_data, epochs=5)
model = image_classifier.create(
    train_data,
    epochs=epoch,
    model_spec=spec,
    train_whole_model=False,
    validation_data=test_data,
    batch_size=bsize,
    dropout_rate=0.3,
    # learning_rate=0.01,
    # use_augmentation=True,
)


print("exporting saved_model..")
model.export(export_dir=image_path + "model_saved_model", export_format=ExportFormat.SAVED_MODEL)
print("exporting label..")
model.export(export_dir=image_path + "model_label", export_format=ExportFormat.LABEL)

print("exporting model..")
model.export(export_dir=image_path + "model_default")


loss, accuracy = model.evaluate(test_data)

print(f"loss, accuracy: {loss}, {accuracy}")
