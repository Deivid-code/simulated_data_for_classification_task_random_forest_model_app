import streamlit as st
import pickle
import numpy as np

# Cargar el modelo
with open('simulated_data_for_classification_task_random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

def main():
    # Configurar la aplicación
    st.title('Predicción con Random Forest')

    # Crear entradas para las variables
    var1 = st.number_input('Ingrese el valor para la Variable 1', min_value=0.0, value=0.0)
    var2 = st.number_input('Ingrese el valor para la Variable 2', min_value=0.0, value=0.0)

    if st.button('Realizar Predicción'):
        # Preparar los datos para la predicción
        data = np.array([[var1, var2]])
        
        # Hacer la predicción
        prediction = model.predict(data)
        
        # Mostrar el resultado
        st.write(f'La predicción del modelo es: {prediction[0]}')

if __name__ == '__main__':
    main()
