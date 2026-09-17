import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def cargar_y_limpiar_datos():
    df = pd.read_csv('train.csv')
    
    # Rellenar valores nulos de todas las columnas
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Convertir variables categóricas a numéricas
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    X = df[features]
    y = df['Survived']
    return X, y

def entrenar_modelo(X, y):
    print("--- ENTRENANDO MODELO DE MACHINE LEARNING ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = LogisticRegression(max_iter=500)
    modelo.fit(X_train, y_train)
    
    predicciones = modelo.predict(X_test)
    precision = accuracy_score(y_test, predicciones)
    
    print(f"\nExactitud del modelo (Accuracy): {precision:.4f}\n")
    print("Reporte de Clasificacion:")
    print(classification_report(y_test, predicciones))

if __name__ == "__main__":
    X, y = cargar_y_limpiar_datos()
    entrenar_modelo(X, y)