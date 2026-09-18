import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cargar_y_explorar():
    df = pd.read_csv('train.csv')
    
    print("=== 1. EXPLORACIÓN INICIAL DE LOS DATOS ===")
    print(f"Número de pasajeros (filas): {df.shape[0]}")
    print(f"Número de columnas: {df.shape[1]}")
    print("\nVariables disponibles y tipos de datos:")
    print(df.dtypes)
    
    print("\nValores faltantes por columna:")
    print(df.isnull().sum())
    
    print(f"\nRegistros duplicados: {df.duplicated().sum()}")
    
    print("\nEstadísticas descriptivas (variables numéricas):")
    print(df.describe())
    
    return df

def tratar_faltantes_y_transformar(df):
    print("\n=== 2. TRATAMIENTO DE FALTANTES Y TRANSFORMACIONES ===")
    
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    df['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    df.drop(columns=['Cabin'], inplace=True)
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    bins = [0, 12, 25, 60, 120]
    labels = ['Niño', 'Joven', 'Adulto', 'Adulto mayor']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    
    print("Tratamiento y creación de variables completados.")
    return df

def realizar_analisis(df):
    print("\n=== 3. ANÁLISIS DE DATOS ===")
    
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
    print("\n=== 4. GENERANDO VISUALIZACIONES ===")
    
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

def imprimir_conclusiones():
    print("\n=== 5. CONCLUSIONES DEL ANÁLISIS ===")
    print("1. El sesgo de género es determinante: Las mujeres tuvieron mayor supervivencia.")
    print("2. La clase socioeconómica importó: Mayor supervivencia en 1ª clase.")
    print("3. Prioridad de rescate a los niños.")

if __name__ == "__main__":
    df = cargar_y_explorar()
    df = tratar_faltantes_y_transformar(df)
    realizar_analisis(df)
    generar_visualizaciones(df)
    imprimir_conclusiones()