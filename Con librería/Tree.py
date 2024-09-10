from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

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


# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Árbol de decisón
# Uso entropía para mejorar la calidad de cada nodo y así no tener un árbol de mucha profundidad, esto para evitar el overfitting
arbol = DecisionTreeClassifier(criterion='entropy',max_depth=6)
arbol.fit(X_train, y_train)
y_pred = arbol.predict(X_test)


# Evalúo el rendimiento del modelo con accuracy, matriz de confusión y reporte de clasificación
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo de prueba: {accuracy}')

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("Informe de clasificación:")
print(classification_report(y_test, y_pred))
