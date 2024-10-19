import requests
import pandas as pd

# Función para obtener datos de una API
def obtener_datos_api(url):
    respuesta = requests.get(url)
    if respuesta.status_code == 200:
        return respuesta.json()  # Suponemos que la respuesta está en formato JSON
    else:
        raise Exception(f"Error al obtener datos: {respuesta.status_code}")

# URL de la API
url_api = 'nomics.com'  # Reemplaza con la URL real de la API
datos = obtener_datos_api(url_api)

# Convertir a DataFrame de pandas
df = pd.DataFrame(datos)
print(df.head())  # Muestra las primeras filas de los datos
# Preprocesamiento de datos
df = df.dropna()  # Eliminar filas con valores nulos
df_normalizado = (df - df.mean()) / df.std()  # Normalización
from sklearn.decomposition import FastICA

# Definir el número de componentes que deseamos
n_componentes = 2  # Cambia esto según tus necesidades

# Aplicar ICA
ica = FastICA(n_components=n_componentes)
componentes_independientes = ica.fit_transform(df_normalizado)

# Convertir los resultados a un DataFrame
df_ica = pd.DataFrame(componentes_independientes, columns=[f'Componente_{i+1}' for i in range(n_componentes)])
print(df_ica.head())  # Muestra las primeras filas de los componentes independientes
import matplotlib.pyplot as plt

# Visualizar los componentes
plt.figure(figsize=(10, 5))
plt.scatter(df_ica['Componente_1'], df_ica['Componente_2'])
plt.title('Componentes Independientes')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.grid()
plt.show()
