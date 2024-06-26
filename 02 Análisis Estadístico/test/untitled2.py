# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rsxNVmai3HXgwPcLeywiOKw-oJRwCgAr
"""

import pandas as pd
import numpy as np

# Establecer la cantidad de datos a generar
cantidad_datos = 100

# Definir funciones para generar datos simulados
def generar_edad():
    return np.random.randint(18, 80, cantidad_datos)

# Generar datos simulados
edad = generar_edad()
peso = generar_peso(edad)
altura = generar_altura()
circunferencia_cintura = generar_circunferencia_cintura()
circunferencia_cadera = generar_circunferencia_cadera()
porcentaje_grasa_corporal = generar_porcentaje_grasa_corporal()
historial_medico_familiar = generar_historial_medico_familiar()
nivel_actividad_fisica = generar_nivel_actividad_fisica()
habitos_alimenticios = generar_habitos_alimenticios()
horas_sueño_noche = generar_horas_sueño_noche()
nivel_estres_percibido = generar_nivel_estres_percibido()
consumo_agua_diario = generar_consumo_agua_diario()
consumo_alcohol = generar_consumo_alcohol()
consumo_tabaco = generar_consumo_tabaco()
consumo_cafeina = generar_consumo_cafeina()
enfermedades_cronicas = generar_enfermedades_cronicas()
medicamentos_actuales = generar_medicamentos_actuales()
metas_perdida_peso = generar_metas_perdida_peso()
frecuencia_cardiaca_reposo = generar_frecuencia_cardiaca_reposo()
presion_arterial_sistolica = generar_presion_arterial_sistolica()
presion_arterial_diastolica = generar_presion_arterial_diastolica()
ldl, hdl, trigliceridos = generar_niveles_colesterol()
ayunas, postprandial = generar_niveles_glucosa_sangre()
sensibilidad_alimentos = generar_sensibilidad_alimentos()
nivel_satisfaccion_dieta_actual = generar_nivel_satisfaccion_dieta_actual()
cumplimiento_plan_nutricional = generar_cumplimiento_plan_nutricional()
actividades_fisicas_realizadas = generar_actividades_fisicas_realizadas()
consumo_frutas_verduras = generar_consumo_frutas_verduras()
nivel_conocimiento_nutricion = generar_nivel_conocimiento_nutricion()

# Crear un diccionario con los datos generados
data = {
    'Edad': edad,
    'Peso': peso,
    'Altura': altura,
    'IMC': calcular_imc(peso, altura),
    'Circunferencia de cintura': circunferencia_cintura,
    'Circunferencia de cadera': circunferencia_cadera,
    'Porcentaje de grasa corporal': porcentaje_grasa_corporal,
    'Historial médico familiar': historial_medico_familiar,
    'Nivel de actividad física': nivel_actividad_fisica,
    'Hábitos alimenticios': habitos_alimenticios,
    'Horas de sueño por noche': horas_sueño_noche,
    'Nivel de estrés percibido': nivel_estres_percibido,
    'Consumo de agua diario': consumo_agua_diario,
    'Consumo de alcohol': consumo_alcohol,
    'Consumo de tabaco': consumo_tabaco,
    'Consumo de cafeína': consumo_cafeina,
    'Enfermedades crónicas': enfermedades_cronicas,
    'Medicamentos actuales': medicamentos_actuales,
    'Metas de pérdida de peso': metas_perdida_peso,
    'Frecuencia cardíaca en reposo': frecuencia_cardiaca_reposo,
    'Presión arterial sistólica': presion_arterial_sistolica,
    'Presión arterial diastólica': presion_arterial_diastolica,
    'Niveles de colesterol (LDL)': ldl,
    'Niveles de colesterol (HDL)': hdl,
    'Niveles de colesterol (Triglicéridos)': trigliceridos,
    'Niveles de glucosa en sangre (Ayunas)': ayunas,
    'Niveles de glucosa en sangre (Postprandial)': postprandial,
    'Sensibilidad a ciertos alimentos': sensibilidad_alimentos,
    'Nivel de satisfacción con la dieta actual': nivel_satisfaccion_dieta_actual,
    'Cumplimiento con el plan nutricional': cumplimiento_plan_nutricional,
    'Actividades físicas realizadas': actividades_fisicas_realizadas,
    'Consumo de frutas y verduras': consumo_frutas_verduras,
    'Nivel de conocimiento sobre nutrición': nivel_conocimiento_nutricion
}

# Crear un DataFrame de Pandas
df = pd.DataFrame(data)

import pandas as pd
import numpy as np

# Establecer la cantidad de datos a generar
cantidad_datos = 100

# Definir funciones para generar datos simulados
def generar_edad():
    return np.random.randint(18, 80, cantidad_datos)

def generar_peso(edad):
    # Para simular casos de obesidad en ciertas edades
    sobrepeso = np.random.choice([0, 1], cantidad_datos, p=[0.9, 0.1])  # 10% de casos de obesidad
    peso_base = np.random.normal(70, 15, cantidad_datos)  # Distribución normal con media de 70 kg y desviación estándar de 15 kg
    peso = peso_base + sobrepeso * 20  # Añadir 20 kg en caso de obesidad
    # Ajustar el peso basado en la edad
    peso = np.where((edad >= 40) & (edad <= 60), peso + 5, peso)  # Añadir 5 kg para edades entre 40 y 60
    return peso

def generar_altura():
    # Distribución normal con media de 170 cm y desviación estándar de 10 cm
    return np.random.normal(170, 10, cantidad_datos)

def calcular_imc(peso, altura):
    # Fórmula del Índice de Masa Corporal (IMC): peso (kg) / altura (m)^2
    return peso / ((altura/100) ** 2)

def generar_circunferencia_cintura():
    # Distribución normal con media de 90 cm y desviación estándar de 10 cm
    return np.random.normal(90, 10, cantidad_datos)

def generar_circunferencia_cadera():
    # Distribución normal con media de 100 cm y desviación estándar de 10 cm
    return np.random.normal(100, 10, cantidad_datos)

def generar_porcentaje_grasa_corporal():
    # Distribución normal con media de 25% y desviación estándar de 5%
    return np.random.normal(25, 5, cantidad_datos)

def generar_historial_medico_familiar():
    # Simulación de historial médico familiar
    enfermedades = ['Diabetes', 'Hipertensión', 'Cáncer', 'Enfermedades cardiovasculares']
    return [', '.join(np.random.choice(enfermedades, np.random.randint(0, len(enfermedades)), replace=False)) for _ in range(cantidad_datos)]

def generar_nivel_actividad_fisica():
    return np.random.choice(['Sedentario', 'Ligero', 'Moderado', 'Intenso'], cantidad_datos)

def generar_habitos_alimenticios():
    habitos = ['Vegetariano', 'Omnívoro', 'Vegano', 'Pescetariano', 'Keto', 'Paleo']
    return np.random.choice(habitos, cantidad_datos)

def generar_horas_sueño_noche():
    # Distribución normal con media de 7 horas y desviación estándar de 1 hora
    return np.random.normal(7, 1, cantidad_datos)

def generar_nivel_estres_percibido():
    return np.random.randint(1, 11, cantidad_datos)

def generar_consumo_agua_diario():
    # Distribución normal con media de 2 litros y desviación estándar de 0.5 litros
    return np.random.normal(2, 0.5, cantidad_datos)

def generar_consumo_alcohol():
    # Distribución normal con media de 5 unidades por semana y desviación estándar de 3 unidades por semana
    return np.random.normal(5, 3, cantidad_datos)

def generar_consumo_tabaco():
    # Distribución normal con media de 5 cigarrillos por día y desviación estándar de 2 cigarrillos por día
    return np.random.normal(5, 2, cantidad_datos)

def generar_consumo_cafeina():
    # Distribución normal con media de 200 mg por día y desviación estándar de 100 mg por día
    return np.random.normal(200, 100, cantidad_datos)

def generar_enfermedades_cronicas():
    enfermedades = ['Diabetes', 'Hipertensión', 'Cáncer', 'Enfermedades cardiovasculares', 'Enfermedad renal crónica', 'Enfermedad pulmonar crónica']
    return [', '.join(np.random.choice(enfermedades, np.random.randint(0, len(enfermedades)), replace=False)) for _ in range(cantidad_datos)]

def generar_medicamentos_actuales():
    medicamentos = ['Aspirina', 'Insulina', 'Losartán', 'Atorvastatina', 'Metformina', 'Omeprazol', 'Salbutamol']
    dosis = ['10 mg', '50 mg', '100 mg', '20 mg', '200 UI', '500 mg', '1 tableta']
    return ['{}, {}'.format(np.random.choice(medicamentos), np.random.choice(dosis)) for _ in range(cantidad_datos)]

def generar_metas_perdida_peso():
    # Distribución normal con media de 5 kg y desviación estándar de 3 kg
    return np.random.normal(5, 3, cantidad_datos)

def generar_frecuencia_cardiaca_reposo():
    # Distribución normal con media de 70 latidos por minuto y desviación estándar de 10 latidos por minuto
    return np.random.normal(70, 10, cantidad_datos)

def generar_presion_arterial_sistolica():
    # Distribución normal con media de 120 mmHg y desviación estándar de 10 mmHg
    return np.random.normal(120, 10, cantidad_datos)

def generar_presion_arterial_diastolica():
    # Distribución normal con media de 80 mmHg y desviación estándar de 8 mmHg
    return np.random.normal(80, 8, cantidad_datos)

def generar_niveles_colesterol():
    ldl = np.random.normal(100, 20, cantidad_datos)  # LDL: media de 100 mg/dL, desviación estándar de 20 mg/dL
    hdl = np.random.normal(50, 10, cantidad_datos)  # HDL: media de 50 mg/dL, desviación estándar de 10 mg/dL
    trigliceridos = np.random.normal(150, 30, cantidad_datos)  # Triglicéridos: media de 150 mg/dL, desviación estándar de 30 mg/dL
    return ldl, hdl, trigliceridos

def generar_niveles_glucosa_sangre():
    ayunas = np.random.normal(90, 10, cantidad_datos)  # Niveles de glucosa en ayunas: media de 90 mg/dL, desviación estándar de 10 mg/dL
    postprandial = np.random.normal(120, 20, cantidad_datos)  # Niveles de glucosa postprandial: media de 120 mg/dL, desviación estándar de 20 mg/dL
    return ayunas, postprandial

def generar_sensibilidad_alimentos():
    alimentos = ['Lactosa', 'Gluten', 'Nueces', 'Mariscos', 'Huevo', 'Soja']
    return [', '.join(np.random.choice(alimentos, np.random.randint(0, len(alimentos)), replace=False)) for _ in range(cantidad_datos)]

def generar_nivel_satisfaccion_dieta_actual():
    return np.random.randint(1, 11, cantidad_datos)

def generar_cumplimiento_plan_nutricional():
    return np.random.randint(1, 11, cantidad_datos)

def generar_actividades_fisicas_realizadas():
    actividades = ['Caminar', 'Correr', 'Nadar', 'Bailar', 'Levantamiento de pesas', 'Yoga']
    return [', '.join(np.random.choice(actividades, np.random.randint(1, 4), replace=False)) for _ in range(cantidad_datos)]

def generar_consumo_frutas_verduras():
    # Distribución normal con media de 5 porciones por día y desviación estándar de 2 porciones por día
    return np.random.normal(5, 2, cantidad_datos)

def generar_nivel_conocimiento_nutricion():
    return np.random.choice(['Bajo', 'Medio', 'Alto'], cantidad_datos)

# Generar datos simulados
edad = generar_edad()
peso = generar_peso(edad)
altura = generar_altura()
circunferencia_cintura = generar_circunferencia_cintura()
circunferencia_cadera = generar_circunferencia_cadera()
porcentaje_grasa_corporal = generar_porcentaje_grasa_corporal()
historial_medico_familiar = generar_historial_medico_familiar()
nivel_actividad_fisica = generar_nivel_actividad_fisica()
habitos_alimenticios = generar_habitos_alimenticios()
horas_sueño_noche = generar_horas_sueño_noche()
nivel_estres_percibido = generar_nivel_estres_percibido()
consumo_agua_diario = generar_consumo_agua_diario()
consumo_alcohol = generar_consumo_alcohol()
consumo_tabaco = generar_consumo_tabaco()
consumo_cafeina = generar_consumo_cafeina()
enfermedades_cronicas = generar_enfermedades_cronicas()
medicamentos_actuales = generar_medicamentos_actuales()
metas_perdida_peso = generar_metas_perdida_peso()
frecuencia_cardiaca_reposo = generar_frecuencia_cardiaca_reposo()
presion_arterial_sistolica = generar_presion_arterial_sistolica()
presion_arterial_diastolica = generar_presion_arterial_diastolica()
ldl, hdl, trigliceridos = generar_niveles_colesterol()
ayunas, postprandial = generar_niveles_glucosa_sangre()
sensibilidad_alimentos = generar_sensibilidad_alimentos()
nivel_satisfaccion_dieta_actual = generar_nivel_satisfaccion_dieta_actual()
cumplimiento_plan_nutricional = generar_cumplimiento_plan_nutricional()
actividades_fisicas_realizadas = generar_actividades_fisicas_realizadas()
consumo_frutas_verduras = generar_consumo_frutas_verduras()
nivel_conocimiento_nutricion = generar_nivel_conocimiento_nutricion()

# Crear un diccionario con los datos generados
data = {
    'Edad': edad,
    'Peso': peso,
    'Altura': altura,
    'IMC': calcular_imc(peso, altura),
    'Circunferencia de cintura': circunferencia_cintura,
    'Circunferencia de cadera': circunferencia_cadera,
    'Porcentaje de grasa corporal': porcentaje_grasa_corporal,
    'Historial médico familiar': historial_medico_familiar,
    'Nivel de actividad física': nivel_actividad_fisica,
    'Hábitos alimenticios': habitos_alimenticios,
    'Horas de sueño por noche': horas_sueño_noche,
    'Nivel de estrés percibido': nivel_estres_percibido,
    'Consumo de agua diario': consumo_agua_diario,
    'Consumo de alcohol': consumo_alcohol,
    'Consumo de tabaco': consumo_tabaco,
    'Consumo de cafeína': consumo_cafeina,
    'Enfermedades crónicas': enfermedades_cronicas,
    'Medicamentos actuales': medicamentos_actuales,
    'Metas de pérdida de peso': metas_perdida_peso,
    'Frecuencia cardíaca en reposo': frecuencia_cardiaca_reposo,
    'Presión arterial sistólica': presion_arterial_sistolica,
    'Presión arterial diastólica': presion_arterial_diastolica,
    'Niveles de colesterol (LDL)': ldl,
    'Niveles de colesterol (HDL)': hdl,
    'Niveles de colesterol (Triglicéridos)': trigliceridos,
    'Niveles de glucosa en sangre (Ayunas)': ayunas,
    'Niveles de glucosa en sangre (Postprandial)': postprandial,
    'Sensibilidad a ciertos alimentos': sensibilidad_alimentos,
    'Nivel de satisfacción con la dieta actual': nivel_satisfaccion_dieta_actual,
    'Cumplimiento con el plan nutricional': cumplimiento_plan_nutricional,
    'Actividades físicas realizadas': actividades_fisicas_realizadas,
    'Consumo de frutas y verduras': consumo_frutas_verduras,
    'Nivel de conocimiento sobre nutrición': nivel_conocimiento_nutricion
}

# Crear un DataFrame de Pandas
df = pd.DataFrame(data)
df

df.describe()

# Verificar y corregir valores nulos o inconsistentes
# for columna in df.columns:
#     print(f"Longitud de {columna}: {len(df[columna])}")

# Manejar valores nulos imputándoles NaN
longitud_maxima = max(len(df[columna]) for columna in df.columns)
for columna in df.columns:
    if len(df[columna]) < longitud_maxima:
        df[columna] = df[columna].append(pd.Series([np.nan] * (longitud_maxima - len(df[columna]))), ignore_index=True)

# Verificar nuevamente las longitudes después de imputar NaN
# print("\nDespués de imputar NaN:\n")
# for columna in df.columns:
#     print(f"Longitud de {columna}: {len(df[columna])}")

# Separar el DataFrame en dos basado en la lógica de los datos
datos_con_logica_real = df[['Niveles de colesterol (LDL)', 'Niveles de colesterol (HDL)', 'Niveles de colesterol (Triglicéridos)',
                            'Niveles de glucosa en sangre (Ayunas)', 'Niveles de glucosa en sangre (Postprandial)']]
datos_aleatorios = df.drop(columns=['Niveles de colesterol (LDL)', 'Niveles de colesterol (HDL)', 'Niveles de colesterol (Triglicéridos)',
                                     'Niveles de glucosa en sangre (Ayunas)', 'Niveles de glucosa en sangre (Postprandial)'])

# Explicar los DataFrames resultantes
print("\nDataFrame con lógica real:\n")
print(datos_con_logica_real.head())

print("\nDataFrame con datos aleatorios:\n")
print(datos_aleatorios.head())

df = datos_aleatorios.copy()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Eliminar columnas no numéricas
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Calcular la matriz de correlación
correlation_matrix = df_numeric.corr()

# Crear un mapa de calor de la matriz de correlación
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlación')

# Agregar explicación de los valores de correlación
plt.text(0.5, -0.5, "Correlación fuerte (+0.7 a +1)\nCorrelación moderada (+0.3 a +0.7)\nCorrelación débil (+0.1 a +0.3)\nCorrelación negativa (-0.1 a -0.3)\nCorrelación moderada negativa (-0.3 a -0.7)\nCorrelación fuerte negativa (-0.7 a -1)", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)

plt.show()

import matplotlib.pyplot as plt

# Histograma de distribución de peso
plt.hist(datos_aleatorios['Peso'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Peso (kg)')
plt.ylabel('Frecuencia')
plt.title('Distribución de peso')
plt.grid(True)
plt.show()

df.columns

test = df.drop(columns= ['Sensibilidad a ciertos alimentos',
       'Nivel de satisfacción con la dieta actual',
       'Cumplimiento con el plan nutricional',
       'Actividades físicas realizadas', 'Consumo de frutas y verduras',
       'Nivel de conocimiento sobre nutrición'], axis=1)
test.columns

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# Generar datos simulados y crear un DataFrame
# (Aquí deberías reemplazar este bloque de código con tu código para generar y cargar los datos simulados)

# Definir las características (features) y la variable objetivo
X = test.drop(columns=['Enfermedades crónicas'])  # Características
y = df['Enfermedades crónicas']  # Variable objetivo

X.columns

# Identificar columnas con variables categóricas
categorical_cols = X.select_dtypes(include=['object']).columns
c = X[categorical_cols]

c["Historial médico familiar"].unique()

X_encoded = pd.get_dummies(X, columns=categorical_cols)
X_encoded

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Inicializar el clasificador Random Forest y entrenar el modelo
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_classifier

import json
# Hacer predicciones en el conjunto de prueba
y_pred = rf_classifier.predict(X_test)
# Obtener los parámetros del modelo
model_params = rf_classifier.get_params()

# Guardar los parámetros en un archivo JSON
with open("rf_model_params.json", "w") as json_file:
    json.dump(model_params, json_file)