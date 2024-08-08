import streamlit as st

# Función para realizar cálculos
def calcular(operacion, num1, num2):
    if operacion == "Suma":
        return num1 + num2
    elif operacion == "Resta":
        return num1 - num2
    elif operacion == "Multiplicación":
        return num1 * num2
    elif operacion == "División":
        if num2 != 0:
            return num1 / num2
        else:
            return "Error: División por cero"
    else:
        return "Operación no válida"

# Título de la aplicación
st.title("Calculadora Simple")

# Selección de operación
operacion = st.selectbox("Selecciona la operación", ["Suma", "Resta", "Multiplicación", "División"])

# Entradas de números
num1 = st.number_input("Número 1", value=0)
num2 = st.number_input("Número 2", value=0)

# Botón para calcular
if st.button("Calcular"):
    resultado = calcular(operacion, num1, num2)
    st.write(f"Resultado: {resultado}")