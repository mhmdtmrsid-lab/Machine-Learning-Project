from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def train_logistic_regression(X_train, y_train):
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    print("Training Support Vector Machine...")
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    print("Training Decision Tree...")
    model = DecisionTreeClassifier(random_state=42)
    
    # Optional: Basic hyperparameter tuning
    param_grid = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Decision Tree params: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_neural_network(X_train, y_train, output_classes):
    print("Training Neural Network...")
    input_dim = X_train.shape[1]
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_classes, activation='softmax' if output_classes > 2 else 'sigmoid')
    ])
    
    loss_fn = 'sparse_categorical_crossentropy' if output_classes > 2 else 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=50, validation_split=0.2, 
              callbacks=[early_stop], verbose=1)
    
    return model