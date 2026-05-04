import joblib
import os

def save_sklearn_model(model, filename):
    filepath = os.path.join('../models', filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_sklearn_model(filename):
    filepath = os.path.join('../models', filename)
    return joblib.load(filepath)

def save_nn_model(model, filename):
    filepath = os.path.join('../models', filename)
    model.save(filepath)
    print(f"Neural Network model saved to {filepath}")
