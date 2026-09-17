#import pandas as pd

#def cargar_datos():
    #print("--- CARGANDO DATOS ---")
    #df = pd.read_csv('train.csv')
    #print(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.\n")
    #print("Primeras 5 filas:")
    #print(df.head())
    #return df

#if __name__ == "__main__":
    #cargar_datos()
    import pandas as pd

def cargar_y_limpiar_datos():
    print("--- CARGANDO Y LIMPIANDO DATOS ---")
    df = pd.read_csv('train.csv')

    # Rellenar valores nulos
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Convertir variables categoricas a numericas
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    X = df[features]
    y = df['Survived']

    print("Limpieza de datos completada exitosamente.")
    return X, y

if __name__ == "__main__":
    X, y = cargar_y_limpiar_datos()