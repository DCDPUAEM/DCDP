import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Definir datos simulados para prueba
data = {
    'Edad': [25],
    'Peso': [70],
    'Altura': [170],
    'IMC': [24],
    'Circunferencia de cintura': [90],
    'Circunferencia de cadera': [100],
    'Porcentaje de grasa corporal': [20],
    'Historial médico familiar': ['Diabetes'],
    'Nivel de actividad física': ['Moderado'],
    'Hábitos alimenticios': ['Omnívoro'],
    'Horas de sueño por noche': [7],
    'Nivel de estrés percibido': [5],
    'Consumo de agua diario': [2],
    'Consumo de alcohol': [3],
    'Consumo de tabaco': [0],
    'Consumo de cafeína': [100],
    'Frecuencia cardíaca en reposo': [70],
    'Presión arterial sistólica': [120],
    'Presión arterial diastólica': [80],
    'Nivel de conocimiento sobre nutrición': ['Medio']
}

# Crear un DataFrame con los datos simulados
df = pd.DataFrame(data)

# Inicializar el clasificador Random Forest y entrenar el modelo
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Definir características (features) y variable objetivo
X_train = pd.DataFrame([[25, 70, 170, 24, 90, 100, 20, 7, 5, 2, 3, 0, 100, 70, 120, 80]], columns=df.columns)
y_train = ['Diabetes']

# Entrenar el modelo
rf_classifier.fit(X_train, y_train)

# Definir la aplicación Streamlit
st.title('Aplicación de Predicción de Enfermedades Crónicas')
st.write('Esta aplicación utiliza un modelo de Random Forest para predecir la presencia de enfermedades crónicas.')

# Mostrar la entrada de usuario y realizar predicciones
st.subheader('Realizar Predicciones Personalizadas')

# Interfaz de usuario para ingresar los datos
edad = st.slider('Edad', min_value=18, max_value=80, value=25)
peso = st.slider('Peso (kg)', min_value=40, max_value=150, value=70)
altura = st.slider('Altura (cm)', min_value=100, max_value=250, value=170)
imc = st.slider('IMC', min_value=10, max_value=40, value=24)
cintura = st.slider('Circunferencia de cintura (cm)', min_value=60, max_value=120, value=90)
cadera = st.slider('Circunferencia de cadera (cm)', min_value=60, max_value=150, value=100)
grasa_corporal = st.slider('Porcentaje de grasa corporal', min_value=5, max_value=40, value=20)
sueño = st.slider('Horas de sueño por noche', min_value=3, max_value=12, value=7)
estres = st.slider('Nivel de estrés percibido', min_value=1, max_value=10, value=5)
agua_diario = st.slider('Consumo de agua diario (litros)', min_value=0.5, max_value=5.0, value=2.0)
alcohol = st.slider('Consumo de alcohol (unidades por semana)', min_value=0, max_value=20, value=3)
tabaco = st.slider('Consumo de tabaco (cigarrillos por día)', min_value=0, max_value=20, value=0)
cafeina = st.slider('Consumo de cafeína (mg por día)', min_value=0, max_value=500, value=100)
frecuencia_cardiaca = st.slider('Frecuencia cardíaca en reposo (latidos por minuto)', min_value=40, max_value=120, value=70)
presion_sistolica = st.slider('Presión arterial sistólica (mmHg)', min_value=90, max_value=180, value=120)
presion_diastolica = st.slider('Presión arterial diastólica (mmHg)', min_value=50, max_value=120, value=80)
conocimiento_nutricion = st.selectbox('Nivel de conocimiento sobre nutrición', ['Bajo', 'Medio', 'Alto'])

# Realizar predicción cuando se presiona el botón
if st.button('Realizar Predicción'):
    user_input = {
        'Edad': [edad],
        'Peso': [peso],
        'Altura': [altura],
        'IMC': [imc],
        'Circunferencia de cintura': [cintura],
        'Circunferencia de cadera': [cadera],
        'Porcentaje de grasa corporal': [grasa_corporal],
        'Horas de sueño por noche': [sueño],
        'Nivel de estrés percibido': [estres],
        'Consumo de agua diario': [agua_diario],
        'Consumo de alcohol': [alcohol],
        'Consumo de tabaco': [tabaco],
        'Consumo de cafeína': [cafeina],
        'Frecuencia cardíaca en reposo': [frecuencia_cardiaca],
        'Presión arterial sistólica': [presion_sistolica],
        'Presión arterial diastólica': [presion_diastolica],
        'Nivel de conocimiento sobre nutrición': [conocimiento_nutricion]
    }
    user_input_df = pd.DataFrame(user_input)

    # Realizar la predicción
    prediction = rf_classifier.predict(user_input_df)
    st.write('La predicción es:', prediction[0])
