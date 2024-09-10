import numpy as np
import pandas as pd

df = pd.read_csv('age_predictions_cleaned.csv')


X = df.drop(['age_group'], axis=1) 
y = df['age_group']

# Dividir datos en validation, train y test
def data_split(X, y, test_size, val_size, random_state):
    np.random.seed(random_state)

    # Mezclar los datos
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # Tamaños
    df_size = X.shape[0]
    test_end = int(df_size * test_size)
    val_end = test_end + int(df_size * val_size)

    # Dividir datos
    X_test = X[:test_end]
    y_test = y[:test_end]
    X_val = X[test_end:val_end]
    y_val = y[test_end:val_end]
    X_train = X[val_end:]
    y_train = y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test

# Ajustamos las variables para el data split
test_size = 0.2
val_size = 0.14
random_state = 42

X_train, X_val, X_test, y_train, y_val, y_test = data_split(X, y, test_size, val_size, random_state)

# Función sigmoidal
def sigmoid_func(z):
    return 1/(1 + np.exp(-z))

# Función de costos
def cost_func(X, y, theta):
    m = X.shape[0]
    h = sigmoid_func(np.dot(X, theta))
    return -(1/m)*(np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)))

# Gradient descent
def gradients(X, y, theta):
    m = X.shape[0]
    h = sigmoid_func(np.dot(X, theta))
    return (1 / m) * np.dot(X.T, (h - y))

def gradient_descent(X, y, theta, alpha, num_iters):
    for _ in range(num_iters):
        grad = gradients(X, y, theta)
        theta = theta - alpha * grad
    return theta

# Hacemos predicciones con la función sigmoidal
def predict(X, theta):
    probabilities = sigmoid_func(np.dot(X, theta))
    predictions = probabilities >= 0.5
    return predictions.astype(int)


# Inicialización de parámetros
theta = np.zeros(X.shape[1])
alpha = 0.01
num_iters = 3000


# Ejecutar el gradiente descendente
theta_optimized = gradient_descent(X, y, theta, alpha, num_iters)


# Evaluamos con datos de validación y de prueba
y_val_pred = predict(X_val, theta_optimized)
y_val_true = y_val

y_test_pred = predict(X_test, theta_optimized)
y_test_true = y_test


# Evaluamos los resultados con accuracy, recall y f1
def model_evaluation(y_pred, y_true):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 0) & (y_true == 1))
    fn = np.sum((y_pred == 1) & (y_true == 0))

    acc = tp / (tp + fp) 
    re = tp / (tp + fn)
    f1 = 2 * (acc * re) / (acc + re) 

    return acc, re, f1, [tp, tn, fp, fn]

acc_val, re_val, f1_val, CM_val = model_evaluation(y_val_true, y_val_pred)
acc_test, re_test, f1_test, CM_test = model_evaluation(y_test_true, y_test_pred)

print(f"Accuracy (Validation): {acc_val}")
print(f"Recall (Validation): {re_val}")
print(f"F1 (Validation): {f1_val}")
print(f"Confusion Matrix (Validation): VP={CM_val[0]}, VN={CM_val[1]}, FP={CM_val[2]}, FN={CM_val[3]}")

print(f"Accuracy (Test): {acc_test}")
print(f"Recall (Test): {re_test}")
print(f"F1 (Test): {f1_test}")
print(f"Confusion Matrix (Test): VP={CM_test[0]}, VN={CM_test[1]}, FP={CM_test[2]}, FN={CM_test[3]}")