import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cargar_y_explorar():
    df = pd.read_csv('train.csv')
    
    print("1. exploracion inicial de los datos (eda)")
    
    filas, columnas = df.shape
    print(f"numero total de pasajeros (filas): {filas}")
    print(f"numero total de columnas (variables): {columnas}")
    
    print("\nvariables disponibles y sus tipos de datos:")
    print(df.dtypes)
    
    print("\ncantidad de valores faltantes por columna:")
    print(df.isnull().sum())
    
    duplicados = df.duplicated().sum()
    print(f"\nnumero de registros duplicados en el dataset: {duplicados}")
    
    print("\nestadisticas descriptivas (variables numericas):")
    print(df.describe())
    
    return df

def tratar_faltantes_y_transformar(df):
    print("\n2. tratamiento de faltantes y transformacion")
    
    mediana_edad = df['Age'].median()
    df['Age'] = df['Age'].fillna(mediana_edad)
    print(f"-> 'Age': imputados valores faltantes con la mediana ({mediana_edad} años).")
    
    moda_embarked = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(moda_embarked)
    print(f"-> 'Embarked': imputados valores faltantes con la moda ('{moda_embarked}').")
    
    if df['Fare'].isnull().sum() > 0:
        mediana_fare = df['Fare'].median()
        df['Fare'] = df['Fare'].fillna(mediana_fare)
        print(f"-> 'Fare': imputado con la mediana ({mediana_fare}).")
    
    df['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    df.drop(columns=['Cabin'], inplace=True)
    print("-> 'Cabin': se transformo en la variable binaria 'Has_Cabin' (1=si, 0=no) y se elimino la columna original.")
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    print("-> nueva variable 'FamilySize' creada: SibSp + Parch + 1.")
    
    bins = [0, 12, 25, 60, 120]
    labels = ['niño', 'joven', 'adulto', 'adulto mayor']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    print("-> nueva variable 'AgeGroup' creada con los rangos: niño (<12), joven (12-24), adulto (25-59), adulto mayor (60+).")
    
    return df

def realizar_analisis(df):
    print("\n3. analisis de datos y preguntas clave")
    
    supervivencia_total = df['Survived'].mean() * 100
    print(f"1. ¿que porcentaje de pasajeros sobrevivio?: {supervivencia_total:.2f}%")
    
    sup_sexo = df.groupby('Sex')['Survived'].mean() * 100
    print("\n2. tasa de supervivencia por sexo (%):")
    print(sup_sexo.round(2))
    
    sup_clase = df.groupby('Pclass')['Survived'].mean() * 100
    print("\n3. tasa de supervivencia por clase de boleto (%):")
    print(sup_clase.round(2))
    
    sup_edad = df.groupby('AgeGroup', observed=False)['Survived'].mean() * 100
    print("\n4. tasa de supervivencia por grupo de edad (%):")
    print(sup_edad.round(2))
    
    df['Is_Alone'] = (df['FamilySize'] == 1).astype(int)
    sup_solo = df.groupby('Is_Alone')['Survived'].mean() * 100
    print("\n5. tasa de supervivencia: solo (1) vs acompañado (0) (%):")
    print(sup_solo.round(2))

def generar_visualizaciones(df):
    print("\n4. generando y guardando visualizaciones")
    
    plt.figure(figsize=(6, 4))
    df.groupby('Sex')['Survived'].mean().plot(kind='bar', color=['pink', 'steelblue'])
    plt.title('tasa de supervivencia por sexo')
    plt.ylabel('proporcion de supervivencia (0 a 1)')
    plt.xlabel('sexo')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('supervivencia_sexo.png')
    plt.close()
    
    plt.figure(figsize=(6, 4))
    df.groupby('Pclass')['Survived'].mean().plot(kind='bar', color='seagreen')
    plt.title('tasa de supervivencia por clase de pasajero')
    plt.ylabel('proporcion de supervivencia (0 a 1)')
    plt.xlabel('clase (1ª, 2ª, 3ª)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('supervivencia_clase.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    df.boxplot(column='Fare', by='Survived')
    plt.title('relacion entre tarifa pagada y supervivencia')
    plt.suptitle('')
    plt.ylabel('tarifa pagada ($)')
    plt.xlabel('sobrevivio (0 = no, 1 = si)')
    plt.tight_layout()
    plt.savefig('tarifa_vs_supervivencia.png')
    plt.close()

    print("graficas guardadas correctamente como:")
    print(" - 'supervivencia_sexo.png'")
    print(" - 'supervivencia_clase.png'")
    print(" - 'tarifa_vs_supervivencia.png'")

def imprimir_conclusiones():
    print("\n5. conclusiones del analisis")
    print("1. regla de 'mujeres y niños primero': la tasa de supervivencia de las mujeres (~74%) fue abrumadoramente superior a la de los hombres (~19%). asimismo, el grupo de 'niños' tuvo la mayor probabilidad de salvamento dentro de las edades.")
    print("2. desigualdad por clase social: los pasajeros de 1ª clase tuvieron la mayor tasa de supervivencia (~63%), mientras que en 3ª clase fue sustancialmente menor (~24%), mostrando que la ubicacion y el costo del boleto influyeron directamente en el acceso a los botes.")
    print("3. estructura familiar: viajar acompañado (FamilySize > 1) incremento ligeramente las probabilidades de sobrevivir en comparacion con viajar completamente solo.")

if __name__ == "__main__":
    df = cargar_y_explorar()
    df = tratar_faltantes_y_transformar(df)
    realizar_analisis(df)
    generar_visualizaciones(df)
    imprimir_conclusiones()