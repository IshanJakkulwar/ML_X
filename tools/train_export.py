#!/usr/bin/env python3
"""
train_export.py

Train a simple MLP on CSV data and export:
 - Keras model.h5
 - TFLite flatbuffer (model.tflite)
 - C header with weight arrays for ml_addon (nn_weights.h)

CSV expected: features...,targets...
"""
import numpy as np
import tensorflow as tf
import sys
import os

def load_csv(path):
    data = np.loadtxt(path, delimiter=',')
    X = data[:, :-2]  # assume last 2 cols are targets (adjust)
    Y = data[:, -2:]
    return X, Y

def make_model(input_dim, output_dim, hidden=16):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(hidden, activation='tanh')(inputs)
    outputs = tf.keras.layers.Dense(output_dim, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def export_header(weights, biases, sizes, acts, out_path):
    with open(out_path, 'w') as f:
        f.write('#pragma once\n#include <vector>\n\n')
        f.write('static std::vector<std::vector<float>> ML_ADDON_W = {\n')
        for arr in weights:
            f.write('  { ' + ', '.join(f'{x:.8f}f' for x in arr.flatten()) + ' },\n')
        f.write('};\n\n')
        f.write('static std::vector<std::vector<float>> ML_ADDON_B = {\n')
        for arr in biases:
            f.write('  { ' + ', '.join(f'{x:.8f}f' for x in arr.flatten()) + ' },\n')
        f.write('};\n\n')
        f.write('static std::vector<size_t> ML_ADDON_SIZES = { ' + ', '.join(str(s) for s in sizes) + ' };\n')
        f.write('static std::vector<const char*> ML_ADDON_ACTS = { ' + ', '.join(f"\"{a}\"" for a in acts) + ' };\n')

def main():
    if len(sys.argv) < 3:
        print("Usage: train_export.py data.csv out_dir")
        return
    data_csv = sys.argv[1]; out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    X, Y = load_csv(data_csv)
    in_dim = X.shape[1]; out_dim = Y.shape[1]
    model = make_model(in_dim, out_dim, hidden=16)
    model.fit(X, Y, epochs=200, batch_size=32, verbose=2)
    model_path = os.path.join(out_dir, 'model.h5')
    model.save(model_path)
    # export tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(os.path.join(out_dir, 'model.tflite'), 'wb') as f:
        f.write(tflite_model)
    # get weights/biases in required layout (out * in) per layer
    weights = []
    biases = []
    sizes = [in_dim]
    acts = []
    for layer in model.layers:
        if len(layer.get_weights()) == 0: continue
        W, B = layer.get_weights()  # W: (in, out)
        W_rm = W.T  # (out, in)
        weights.append(W_rm)
        biases.append(B)
        sizes.append(W_rm.shape[0])
        acts.append(layer.activation.__name__)
    export_header(weights, biases, sizes, acts, os.path.join(out_dir, 'nn_weights.h'))
    print("Exported to", out_dir)

if __name__ == "__main__":
    main()
