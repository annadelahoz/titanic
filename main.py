import pandas as pd

def cargar_datos():
    print("--- CARGANDO DATOS ---")
    df = pd.read_csv('train.csv')
    print(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.\n")
    print("Primeras 5 filas:")
    print(df.head())
    return df

if __name__ == "__main__":
    cargar_datos()