from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

# Cargo mis datos
df = pd.read_csv('dataset.csv') 

# Imputo síntomas faltantes en el dataframe (Se coloca "No respondió" ya que no afecta en el modelo)
imputer = SimpleImputer(strategy='constant', fill_value='No Respondió')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


# Codifico los síntomas (variables independientes)
label_encoders = {}
for column in df_imputed.drop(['Disease'], axis=1): 
    le = LabelEncoder()
    df_imputed[column] = le.fit_transform(df_imputed[column].astype(str)) 
    label_encoders[column] = le


# Codifico enfermedad (variable independiente)
le_target = LabelEncoder()
df_imputed['Disease'] = le_target.fit_transform(df_imputed['Disease'].astype(str))


# Divido el dataset en X y
X = df_imputed.drop(['Disease'], axis=1) 
y = df_imputed['Disease']


# Dividir los datos en conjunto de entrenamiento, validación y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# GridSearchCV para encontrar los mejores parámetros
arbol = DecisionTreeClassifier(criterion='entropy',max_depth=6)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Árbol de decisón
# Uso entropía para mejorar la calidad de cada nodo y así no tener un árbol de mucha profundidad, esto para evitar el overfitting
grid_search = GridSearchCV(estimator=arbol, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred_val = best_model.predict(X_val)


# Evalúo el rendimiento del modelo con accuracy, matriz de confusión y reporte de clasificación
# Uso mis datos de validación
accuracy_val = accuracy_score(y_val, y_pred_val)
print(f'Precisión del modelo (Validación): {accuracy_val}')

mc_val = confusion_matrix(y_val, y_pred_val)
print("Matriz de confusión (Validación):")
print(mc_val)

report_val = pd.DataFrame(classification_report(y_val, y_pred_val, output_dict=True)).transpose()
print("Informe de clasificación (Validación):")
print(report_val)


# Uso mis datos de prueba
y_pred = best_model.predict(X_test)

accuracy_test = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo (Prueba): {accuracy_test}')

mc_test = confusion_matrix(y_test, y_pred)
print("Matriz de confusión (Prueba):")
print(mc_test)

report_test = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
print("Informe de clasificación (Prueba):")
print(report_test)


# Graficar la importancia de las características
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Importancia de las características")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.savefig("importancia de las características.png")

df_cm_val = pd.DataFrame(mc_val, index=[i for i in range(len(mc_val))], columns=[i for i in range(len(mc_val))])
df_cm_test = pd.DataFrame(mc_test, index=[i for i in range(len(mc_test))], columns=[i for i in range(len(mc_test))])

# Graficar la matriz de confusión
plt.figure(figsize=(10,7))
sns.heatmap(df_cm_val, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
plt.savefig('hm_valid.png')

# Graficar la matriz de confusión
plt.figure(figsize=(10,7))
sns.heatmap(df_cm_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
plt.savefig('hm_test.png')

# Gráficas de barras para comparar métricas de datos de prueba y validación
plt.figure(figsize=(10,7))
bars = plt.bar(["Accuracy validation", "Accuracy test", "Recall test", "Recall validation", "F1-score test", "F1-score validation"],[accuracy_val, accuracy_test, np.mean(report_test['recall']), np.mean(report_val['recall']), np.mean(report_test['f1-score']), np.mean(report_val['f1-score'])], color=["blue", "blue", "purple", "purple", "red", "red"])

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')

plt.savefig('bar.png')


# Gráfica de sepración de datos
plt.figure(figsize=(10,7))
plt.bar(["Entrenamiento", "Prueba", "Validación"], [X_train.shape[0], X_test.shape[0], X_val.shape[0]])
plt.savefig('Datos.png')