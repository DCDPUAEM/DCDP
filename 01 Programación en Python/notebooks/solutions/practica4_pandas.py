# -*- coding: utf-8 -*-
"""Practica4_Pandas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/DCDPUAEM/DCDP/blob/main/01%20Programaci%C3%B3n%20en%20Python/notebooks/exercises/Practica4_Pandas.ipynb

<a href="https://colab.research.google.com/github/DCDPUAEM/DCDP/blob/main/01%20Programaci%C3%B3n%20en%20Python/notebooks/exercises/Practica4_Pandas.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""### __Data set de vinos del mundo__

Esta es una versión reducida de la base de datos [winemag-data](https://gist.github.com/clairehq/79acab35be50eaf1c383948ed3fd1129), que contiene una reseña en inglés sobre una gran cantidad de vinos del mundo.

### __Descripción de los campos__

 - **country**: El país de donde proviene el vino
 - **description** : Algunas frases de un sommelier que describen el sabor, olor, apariencia, sensación, etc. del vino.
 - **designation**: La denominación. El viñedo dentro de la bodega de donde proceden las uvas que elaboraron el vino.
 - **points**: la cantidad de puntos que WineEnthusiast calificó al vino en una escala del 1 al 100 (aunque dicen que solo publican reseñas de vinos con una puntuación> = 80).
 - **price**: El costo de una botella de vino.
 - **province**: La provincia o estado de donde proviene el vino
 - **region_1**: el área de cultivo de vino en una provincia o estado (es decir, Napa)
 - **region_2**: a veces hay regiones más específicas, especificadas dentro de un área de cultivo del vino (es decir, Rutherford dentro del Valle de Napa), pero este valor a veces puede estar en blanco.
 - **taster_name**: nombre de la persona que probó y revisó el vino.
 - **taster_twitter_handle**: identificador de Twitter para la persona que probó y revisó el vino.
 - **title**: el título de la reseña de vinos, que a menudo contiene la cosecha si está interesado en extraer esa característica.
 - **variety**: la variedad: el tipo de uva utilizada para elaborar el vino (es decir, Pinot Noir).
 - **winery**: la bodega que hizo el vino.

### TEST
"""

# Fetch the dataset using the raw GitHub URL.
!curl --remote-name \
     -H 'Accept: application/vnd.github.v3.raw' \
     --location https://raw.githubusercontent.com/DCDPUAEM/DCDP/main/01%20Programaci%C3%B3n%20en%20Python/data/winemag-data-less.csv

# leemos el dataframe usando read_csv
df = pd.read_csv("winemag-data-less.csv")
print(df.info())
df.head(3)

#Tiremos la columna Unnamed
df.drop('Unnamed: 0',axis=1,inplace=True)
df.head(3)

"""### __Veamos cuantas reseñas de vinos mexicanos tenemos.__

#### &#9758; Construye una nueva Tabla con las reseñas de vinos mexicanos.
- Quédate **sólo** con las siguientes columnas: `['country','winery','variety','description','points','price']`
- Haz que el índice se reinicie en 0.
- Guarda este DataFrame en la variable vinosMX.
"""

"""
SECUENCIA:
1. Ubicar todos los registros de México usando indexación booleana
2. De este DataFrame resultante, extraer solamente las columnas solicitadas.
3. Reiniciar el índice usando reset_index()
4. Tirar (drop) la nueva columna index
5. Asignar el resultado de este proceso a la variable vinosMX
"""
# TU CODIGO
# 1. Ubicar todos los registros de México usando indexación booleana
vinosMX = df[df['country']=='Mexico']

# 2. De este DataFrame resultante, extraer solamente las columnas solicitadas.
vinosMX = vinosMX[['country','winery','variety','description','points','price']]

# 3. Reiniciar el índice usando reset_index()
vinosMX = vinosMX.reset_index()

# 4. Tirar (drop) la nueva columna index
vinosMX = vinosMX.drop('index',axis=1)
vinosMX

"""### __Veamos cuantas reseñas de vinos por país tenemos.__

#### &#9758; Muestra en una gráfica de barras la distribución del número de reseñas por país (_top 10_).

#### Puedes auxiliarte con alguna de estos métodos de Pandas:
 - [pandas.DataFrame.count](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.count.html)
 - [pandas.Series.value_counts](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html)
 - [pandas.Series.index](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.index.html)
 - [pandas.Series.values](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.values.html)
"""

"""
SECUENCIA:
1. Ubicar el nombre de la columna de país
2. Sobre esta columna, obtener la Serie correspondiente
3. Hacer un conteo de los valores únicos sobre esta Serie

4. X en la gráfica de barras son los países (índice de la Serie)
5. Y en la gráfica de barras son los conteos por país (values de la Serie)
6. Usar Seaborn para graficar el diagrama de barras
7. Rotular la gráfica y los ejes
"""
# 1. Ubicar el nombre de la columna de país
# 2. Sobre esta columna, obtener la Serie correspondiente
# 3. Hacer un conteo de los valores únicos sobre esta Serie
conteos=df.country.value_counts()

# 4. X en la gráfica de barras son los países (índice de la Serie)
# 5. Y en la gráfica de barras son los conteos por país (values de la Serie)
x = conteos.index
y = conteos.values

plt.figure(figsize=(16,7))
# Define colors for each bar
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

# Create the bar plot with specified colors
sns.barplot(x=x[:10], y=y[:10], palette="hls")

# En una sola línea:
#sns.barplot(df.country.value_counts().index[:10], df.country.value_counts().values[:10])

plt.xlabel("Países")
plt.ylabel("Número de vinos")
plt.title("Reseñas por país (Top 10)")

plt.show()

"""### __Veamos ahora cuál es el precio promedio por cada país.__

#### &#9758; Muestra en una gráfica de barras el precio promedio por país, en orden descendente (_top 10_).

Puedes consultar:
 - [pandas.DataFrame.groupby](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html)

#### &#9758; ¿Cuál sería el precio promedio por variedad?

🙂 __Escribe la secuencia de pasos que tendrías que realizar.__
"""

df.columns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suponiendo que tienes un DataFrame llamado df con las columnas proporcionadas

# 1. Calcular el precio promedio por país
precio_promedio_por_pais = df.groupby('country')['price'].mean()

# 2. Ordenar los resultados en orden descendente según el precio promedio
precio_promedio_por_pais = precio_promedio_por_pais.sort_values(ascending=False)

# 3. Seleccionar los 10 primeros países
top_10_paises = precio_promedio_por_pais.head(10)

# 4. Graficar los resultados en una gráfica de barras
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_paises.index, y=top_10_paises.values)
plt.xlabel('País')
plt.ylabel('Precio Promedio (Dólares)')
plt.title('Precio Promedio por País (Top 10)')
plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para una mejor legibilidad
plt.show()

# ¿Cuál sería el precio promedio por variedad?
precio_promedio_por_variedad = df.groupby('variety')['price'].mean()
print(precio_promedio_por_variedad)

"""#### Agregando anotaciones a nuestros gráficos

Es posible agregar información a nuestros gráficos, en forma de texto, o dibujos (e.g. flechas, líneas, círculos, etc.). Para ello, se utiliza la anotación (annotate) de ejes (axes). Detallar aquí cómo funciona nos llevaría mucho tiempo. Puedes consultar la documentación en estas ligas:

- [matplotlib.Artist](https://matplotlib.org/3.3.3/api/artist_api.html#matplotlib.artist.Artist)
    - [Artist tutorial](https://matplotlib.org/3.3.3/tutorials/intermediate/artists.html)
- [matplotlib.axes](https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
- [matplotlib.patches.Patch](https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
- [matplotlib.patches.Rectangle](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.Rectangle.html)
- [matplotlib.pyplot.annotate](https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.annotate.html)
"""

#Definimos el tamaño del canvas
plt.figure(figsize=(16,7))

# la variable "ax" (axes) contiene la información  del gráfico de barras.
# En particular, contiene todo lo relativo a los parches (rectángulos) del barplot.
ax = sns.barplot(x=x[:10], y=y[:10])

# Recorremos cada rectángulo
for p in ax.patches:
    ax.annotate("%.2f" % p.get_height(),\
                (p.get_x() + p.get_width() / 2., p.get_height()),\
                ha='center', va='center', fontsize=11, color='gray',\
                xytext=(0, 10),\
                textcoords='offset points')

plt.show()

"""### __Ahora queremos darnos una idea de cuáles podrían ser los países cuyos vinos tienen una mejor razón calidad-precio en promedio.__

#### &#9758; Muestra en una gráfica de barras la razón puntos/precio promedio por país, en orden descendente (_top 10_). Muestra los valores de la razón sobre cada barra.

✋ __Recuerda que si divides entre 0 o Nan obtendrás inf o nan__

🙂 __Escribe la secuencia de pasos que tendrías que realizar.__
"""

plt.figure(figsize=(8,4))
sns.boxplot(x=df.points)
plt.title("Boxplot del puntaje (calidad)")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))

# 1. Calcular la razón calidad/precio por país
promedios = df.groupby('country').apply(lambda x: x['points'].mean() / x['price'].mean())

# 2. Ordenar los resultados en orden descendente según la razón calidad/precio
promedios = promedios.sort_values(ascending=False)

# 3. Seleccionar los 10 primeros países
top_10_paises = promedios.head(10)

# 4. Graficar los resultados en una gráfica de barras
ax = sns.barplot(x=top_10_paises.index, y=top_10_paises.values)
plt.title("Mejor razón calidad/precio por país (Top 10)")
plt.xlabel("País")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Razón Puntos/Precio")

# Añadir etiquetas con los valores en cada barra
for p in ax.patches:
    ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=11, color='gray', xytext=(0, 10),
                 textcoords='offset points')

plt.show()

"""### __¿Qué vinos tienen la mejor puntuación y a qué países pertenecen?__

#### &#9758; Muestra en una gráfica de pastel la proporción de los países que tienen los 20 mejores vinos; es decir, los primeros 20 de mayor puntaje.  
"""

import matplotlib.pyplot as plt

# 1. Ordenar los vinos según su puntuación en orden descendente
top_vinos = df.nlargest(20, 'points')

# 2. Seleccionar los primeros 20 vinos con la mejor puntuación
top_vinos_paises = top_vinos['country']

# 3. Obtener los países a los que pertenecen estos vinos
conteo_paises = top_vinos_paises.value_counts()

# 4. Graficar la proporción de los países en una gráfica de pastel
plt.figure(figsize=(8, 8))
plt.pie(conteo_paises, labels=conteo_paises.index, autopct='%1.1f%%', startangle=140)
plt.title('Proporción de los países con los 20 mejores vinos')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

"""### __¿Cuáles son las 10 variedades de uva más abundantes y en qué proporción?__

#### &#9758; Da una solución utilizando sólo dos líneas de código
"""

#TU CODIGO
#variedades = #<COMPLETA>
variedades = df['variety'].value_counts().head(10)
variedades

"""### __¿Cuáles son las 20 bodegas más mencionadas y en qué proporción?__

#### &#9758; Da una solución utilizando sólo dos líneas de código
"""

#TU CODIGO
bodegas = df['winery'].value_counts().head(10)
bodegas

"""### __Vamos ahora a construir una nueva tabla de información__

#### &#9758; Construye una tabla que muestre país, variedad de uva, bodega, y valores de puntuación y precio.
- Considera las 10 variedades de uva y las 20 bodegas más importantes.
- Haz una tabla con estos datos y sólo los valores de país, bodega, variedad, puntuación y precio correspondientes.
- Agrega una columna con el valor de puntos/precio
- Cambia el nombre de la bodega para que éste incluya su país de origen.
- Ordena los datos por nombre de país, variedad y bodega (orden ascendente).
"""

# Paso 1: Filtrar el DataFrame original
top_variedades = df['variety'].value_counts().head(10).index.tolist()
top_bodegas = df['winery'].value_counts().head(20).index.tolist()
filtered_df = df[df['variety'].isin(top_variedades) & df['winery'].isin(top_bodegas)]

# Paso 2: Seleccionar solo las columnas relevantes
relevant_columns = ['country', 'winery', 'variety', 'points', 'price']
filtered_df = filtered_df[relevant_columns]

# Paso 3: Calcular el valor de puntos/precio
filtered_df['points/price'] = filtered_df['points'] / filtered_df['price']

# Paso 4: Cambiar el nombre de la bodega para que incluya su país de origen
filtered_df['winery'] = filtered_df['country'] + ' - ' + filtered_df['winery']

# Paso 5: Ordenar los datos por nombre de país, variedad y bodega
filtered_df = filtered_df.sort_values(by=['country', 'variety', 'winery'], ascending=True)

# Almacenar el nuevo DataFrame en la variable p_v
p_v = filtered_df.copy()

"""#### &#9758; Observa la relación puntos-precio por país"""

sns.relplot(x="points", y="price", hue="country", col='country',kind="line", data=p_v)

"""#### &#9758; Observa las distribuciones por pares en función de la variedad de uva."""

sns.pairplot(p_v, hue="variety",height=3,palette='rocket')

"""#### &#9758; Observa las distribuciones por pares en función del país.

1.   Elemento de la lista
2.   Elemento de la lista


"""

#TU CODIGO
import seaborn as sns

# Crear las distribuciones por pares en función del país
sns.pairplot(p_v, hue='country')
plt.show()

"""#### &#9758; Observa las distribuciones por pares en función de la bodega."""

#TU CODIGO
import seaborn as sns

# Crear las distribuciones por pares en función de la bodega
sns.pairplot(p_v, hue='winery')
plt.show()

"""#### &#9758; Observa las distribuciones (scatter) de precio por variedad de uva.

> Agregar bloque entrecomillado


"""

#TU CODIGO
import seaborn as sns
import matplotlib.pyplot as plt

# Crear un gráfico de dispersión de precio por variedad de uva
plt.figure(figsize=(10, 6))
sns.scatterplot(data=p_v, x='variety', y='price', alpha=0.7)
plt.xlabel('Variedad de Uva')
plt.ylabel('Precio')
plt.title('Distribución de Precio por Variedad de Uva')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

"""#### &#9758; Observa las distribuciones (scatter) de precio por bodega."""

#TU CODIGO
import seaborn as sns
import matplotlib.pyplot as plt

# Crear un gráfico de dispersión de precio por bodega
plt.figure(figsize=(10, 6))
sns.scatterplot(data=p_v, x='winery', y='price', alpha=0.7)
plt.xlabel('Bodega')
plt.ylabel('Precio')
plt.title('Distribución de Precio por Bodega')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

"""#### &#9758; Observa las distribuciones (scatter) de puntos/precio por bodega."""

#TU CODIGO
import seaborn as sns
import matplotlib.pyplot as plt

# Crear un gráfico de dispersión de puntos/precio por bodega
plt.figure(figsize=(10, 6))
sns.scatterplot(data=p_v, x='winery', y='points/price', alpha=0.7)
plt.xlabel('Bodega')
plt.ylabel('Puntos/Precio')
plt.title('Distribución de Puntos/Precio por Bodega')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

"""#### &#9758; Observa las distribuciones (scatter) de puntos/precio por país."""

#TU CODIGO
import seaborn as sns
import matplotlib.pyplot as plt

# Crear un gráfico de dispersión de puntos/precio por país
plt.figure(figsize=(10, 6))
sns.scatterplot(data=p_v, x='country', y='points/price', alpha=0.7)
plt.xlabel('País')
plt.ylabel('Puntos/Precio')
plt.title('Distribución de Puntos/Precio por País')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

"""### __Agrega los datos de México a esta última tabla de información__

#### &#9758; Une la tabla de vinosMX a la tabla p_v
- Asegúrate de __no agregar__ la columna de descripción
"""

# Suponiendo que 'vinosMX' es el DataFrame que contiene los datos de vinos de México

# Renombrar la columna 'bodega' en 'vinosMX' para que coincida con la columna en 'p_v'
vinosMX = vinosMX.rename(columns={'bodega': 'winery'})

# Eliminar la columna de descripción si existe en 'vinosMX'
if 'description' in vinosMX.columns:
    vinosMX = vinosMX.drop(columns=['description'])

# Unir las tablas utilizando la columna 'country' como clave de unión
df3 = pd.merge(p_v, vinosMX, on='country', how='outer')

# Mostrar la tabla completa
print(df3)

"""#### &#9758; Calcula los valores de points/price para los vinos de México
- TIP: Usa el método apply sobre `df3[['points','price','points/price']]`
"""

# Paso 1: Filtrar el DataFrame para incluir solo los vinos de México
vinos_mexico = df3[df3['country'] == 'Mexico']

# Paso 2: Calcular los valores de points/price
vinos_mexico['points/price'] = vinos_mexico['points_y'] / vinos_mexico['price_y']

# Paso 3: Mostrar el resultado
print(vinos_mexico[['country', 'winery_y', 'variety_y', 'points_y', 'price_y', 'points/price']])

"""#### &#9758; Observa la relación puntos-precio por país"""

import seaborn as sns
import matplotlib.pyplot as plt

# Crear un gráfico de dispersión con líneas de regresión para cada país
sns.lmplot(x="points_x", y="price_x", hue="country", col='country', data=df3, scatter_kws={'alpha':0.7})
plt.show()

"""#### &#9758; Observa las distribuciones por pares en función del país.
- Construye una tabla auxiliar "mx_top", donde los datos de México (en df3) aparezcan al final de la tabla mx_top.
"""

# Separar los datos de México del resto de los países
mx_data = df3[df3['country'] == 'Mexico']
non_mx_data = df3[df3['country'] != 'Mexico']

# Concatenar los datos no mexicanos con los datos de México al final
mx_top = pd.concat([non_mx_data, mx_data])

# Visualizar las distribuciones por pares en función del país
sns.pairplot(mx_top, hue="country", height=3, palette='bright')
plt.show()

"""#### &#9758; Observa las distribuciones (scatter) de puntos por país."""

import seaborn as sns
import matplotlib.pyplot as plt

# Crear un gráfico de dispersión de puntos por país
plt.figure(figsize=(10, 5))
sns.stripplot(x="country", y="points_x", data=df3, dodge=True, palette='deep', marker='*', size=8)
plt.xticks(rotation=45, ha="right")
plt.xlabel('País')
plt.ylabel('Puntos')
plt.title('Distribución de Puntos por País')
plt.tight_layout()
plt.show()

"""#### &#9758; Observa las distribuciones (scatter) de precios por bodega."""

import seaborn as sns
import matplotlib.pyplot as plt

# Crear un gráfico de dispersión de precios por bodega
plt.figure(figsize=(10, 6))
sns.scatterplot(data=mx_top, x='winery_x', y='price_x', alpha=0.7)
plt.xlabel('Bodega')
plt.ylabel('Precio')
plt.title('Distribución de Precios por Bodega')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

"""#### &#9758; Observa las distribuciones (scatter) de puntos/precio por bodega.
- Dibuja una línea que marque el promedio de todos los datos
- Dibuja marcas ubicadando los valores promedio por cada bodega (TIP: usa `groupby` sobre país y bodega para calcular primero los valores promedio)
"""

mx_top.head()

import seaborn as sns
import matplotlib.pyplot as plt

# Calcular los valores promedio por cada bodega
medias = mx_top.groupby(['country', 'winery_x'])['points/price'].mean().reset_index()

# Calcular el promedio de todos los datos
promedio_general = mx_top['points/price'].mean()

# Crear el gráfico de dispersión de puntos/precio por bodega
plt.figure(figsize=(12, 6))
sns.scatterplot(data=mx_top, x='winery_x', y='points/price', alpha=0.7)

# Dibujar una línea que marque el promedio de todos los datos
plt.axhline(y=promedio_general, color='r', linestyle='--', label='Promedio general')

# Dibujar marcas ubicando los valores promedio por cada bodega
for i in range(len(medias)):
    plt.text(i, medias.iloc[i]['points/price'], f"{medias.iloc[i]['points/price']:.2f}", ha='center', va='bottom', fontsize=8, color='blue')

plt.xlabel('Bodega')
plt.ylabel('Puntos/Precio')
plt.title('Distribución de Puntos/Precio por Bodega')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

"""#### &#9758; Si consideramos la proporción de los países con mejor relación puntos/precio, ¿cómo queda México?
- Usa un gráfico de pastel
"""

# TU CODIGO
# Calcular el promedio general de puntos/precio
promedio_general = mx_top['points/price'].mean()

# Contar cuántos vinos de cada país están por encima del promedio general
mejores_vinos_por_pais = mx_top[mx_top['points/price'] > promedio_general]['country'].value_counts()

# Crear un gráfico de pastel para mostrar la proporción de países con mejores vinos
plt.figure(figsize=(8, 8))
plt.pie(mejores_vinos_por_pais, labels=mejores_vinos_por_pais.index, autopct='%1.1f%%', startangle=140)
plt.title('Proporción de países con mejores vinos (puntos/precio)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Calcular el promedio general de puntos/precio
promedio_general = mx_top['points/price'].mean()

# Contar cuántos vinos de México están por encima del promedio general
mejores_vinos_mexico = (mx_top['country'] == 'Mexico') & (mx_top['points/price'] > promedio_general)

# Calcular la proporción de vinos de México con una relación puntos/precio por encima del promedio general
proporcion_mexico = len(mx_top[mejores_vinos_mexico]) / len(mx_top)

# Crear un gráfico de pastel para mostrar la proporción de vinos de México con mejor relación puntos/precio
plt.figure(figsize=(8, 8))
plt.pie([proporcion_mexico, 1 - proporcion_mexico], labels=['México', 'Otros países'], autopct='%1.1f%%', startangle=140)
plt.title('Proporción de vinos de México con mejor relación puntos/precio')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()