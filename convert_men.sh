#!/bin/bash

path_men=/home/terry/code/celebme/celebme_model_202401/faces/men

tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --signature_name=serving_default \
    --saved_model_tags=serve \
    $path_men/model_saved_model/saved_model \
    $path_men/model_tfjs

