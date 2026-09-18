import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cargar_y_explorar():
    df = pd.read_csv('train.csv')
    
<<<<<<< HEAD
    print("1. EXPLORACIÓN INICIAL DE LOS DATOS")
    print(f"Número de pasajeros (filas): {df.shape[0]}")
    print(f"Número de columnas: {df.shape[1]}")
    print("\nVariables disponibles y tipos de datos:")
    print(df.dtypes)
    
    print("\nValores faltantes por columna:")
    print(df.isnull().sum())
=======

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
>>>>>>> df18c3474bc95c9e300d8435a3f7ea137fa32a6a
    
    print(f"\nRegistros duplicados: {df.duplicated().sum()}")
    
    print("\nEstadísticas descriptivas (variables numéricas):")
    print(df.describe())
    
    return df

<<<<<<< HEAD
def tratar_faltantes_y_transformar(df):
    print("\n2. TRATAMIENTO DE FALTANTES Y TRANSFORMACIONES")
=======
def entrenar_modelo(X, y):
    print("ENTRENANDO MODELO DE MACHINE LEARNING")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>>>>>> df18c3474bc95c9e300d8435a3f7ea137fa32a6a
    
    
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
   
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    df['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    df.drop(columns=['Cabin'], inplace=True)
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
   
    bins = [0, 12, 25, 60, 120]
    labels = ['Niño', 'Joven', 'Adulto', 'Adulto mayor']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    
    print("Tratamiento y creación de variables ('FamilySize', 'AgeGroup', 'Has_Cabin') completados.")
    return df

def realizar_analisis(df):
    print("\n3. ANÁLISIS DE DATOS")
    

    supervivencia_gen = (df['Survived'].mean()) * 100
    print(f"1. Porcentaje de supervivencia total: {supervivencia_gen:.2f}%")
  
    sup_sexo = df.groupby('Sex')['Survived'].mean() * 100
    print("\n2. Supervivencia por sexo (%):")
    print(sup_sexo)
    
    sup_clase = df.groupby('Pclass')['Survived'].mean() * 100
    print("\n3. Supervivencia por Clase (%):")
    print(sup_clase)
    
    sup_edad = df.groupby('AgeGroup', observed=False)['Survived'].mean() * 100
    print("\n4. Supervivencia por Grupo de Edad (%):")
    print(sup_edad)

def generar_visualizaciones(df):
    print("\n4. GENERANDO VISUALIZACIONES")
    
    plt.figure(figsize=(6, 4))
    df.groupby('Sex')['Survived'].mean().plot(kind='bar', color=['pink', 'steelblue'])
    plt.title('Tasa de Supervivencia por Sexo')
    plt.ylabel('Proporción de Supervivencia')
    plt.xlabel('Sexo (Female / Male)')
    plt.tight_layout()
    plt.savefig('supervivencia_sexo.png')
    plt.close()
    

    plt.figure(figsize=(6, 4))
    df.groupby('Pclass')['Survived'].mean().plot(kind='bar', color='seagreen')
    plt.title('Tasa de Supervivencia por Clase')
    plt.ylabel('Proporción de Supervivencia')
    plt.xlabel('Clase de Pasajero (1ª, 2ª, 3ª)')
    plt.tight_layout()
    plt.savefig('supervivencia_clase.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    df.boxplot(column='Fare', by='Survived')
    plt.title('Relación entre Tarifa Pagada y Supervivencia')
    plt.suptitle('')
    plt.ylabel('Tarifa ($)')
    plt.xlabel('Sobrevivió (0 = No, 1 = Sí)')
    plt.tight_layout()
    plt.savefig('tarifa_vs_supervivencia.png')
    plt.close()
    
    print("Gráficas guardadas como 'supervivencia_sexo.png', 'supervivencia_clase.png' y 'tarifa_vs_supervivencia.png'.")

def imprimir_conclusiones():
    print("\nCONCLUSIONES DEL ANÁLISIS")
    print("1. El sesgo de género es determinante: Las mujeres tuvieron una probabilidad significativamente mayor de sobrevivir que los hombres.")
    print("2. La clase socioeconómica importó: Los pasajeros de 1ª clase presentaron la tasa más alta de supervivencia, decreciendo hacia la 3ª clase.")
    print("3. La prioridad de rescate favoreció a los niños frente a otros grupos de edad adultas.")

if __name__ == "__main__":
<<<<<<< HEAD
    df = cargar_y_explorar()
    df = tratar_faltantes_y_transformar(df)
    realizar_analisis(df)
    generar_visualizaciones(df)
    imprimir_conclusiones()
=======
    X, y = cargar_y_limpiar_datos()
    entrenar_modelo(X, y)
>>>>>>> df18c3474bc95c9e300d8435a3f7ea137fa32a6a
