import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Cargar el modelo
# with open('simulated_data_for_classification_task_random_forest_model.pkl', 'rb') as file:
#     model = pickle.load(file)

model = joblib.load('model.joblib')

#| echo: false
def generate_data(n_samples=1000, imbalance_ratio=0.8, seed=None):
    # Inicializar el generador de números aleatorios con la semilla, si se      proporciona
    if seed is not None:
        np.random.seed(seed)
    
    n_class1 = int(n_samples * imbalance_ratio)
    n_class0 = n_samples - n_class1
    X_class1 = np.random.randn(n_class1, 2) + np.array([2, 2])
    X_class0 = np.random.randn(n_class0, 2)
    X = np.vstack([X_class1, X_class0])
    y = np.array([1] * n_class1 + [0] * n_class0)
    return X, y

# Ejemplo de uso con semilla
X, y = generate_data(n_samples=1000, imbalance_ratio=0.5, seed=123)
df = pd.DataFrame(X, columns=['x1', 'x2'])
df['y'] = y


def plot_data(X, y):
    plt.figure()
    colores = ['red', 'blue']
    plt.scatter(X[:, 0], X[:, 1], c=[colores[int(i)] for i in y], s=15)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Scatter plot x1 vs x2 \n discriminated by class")

    # Crear objetos PathCollection para etiquetar
    legend_labels = ['Class 0', 'Class 1']
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=color, markersize=10) for label, color in zip(legend_labels, colores)]

    # Agregar leyenda
    plt.legend(handles=legend_elements)

    # Mostrar gráfico
    st.pyplot(plt)

@st.cache_data
def plot_decision_boundary(_model, X, y):
    # Configurar la malla
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predecir la clase en cada punto de la malla
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['red', 'blue']))

    # Graficar los puntos
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=ListedColormap(['red', 'blue']))
    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary and Data Points")

    # Crear leyenda
    handles, labels = scatter.legend_elements()
    plt.legend(handles, ['Class 0', 'Class 1'], loc='best')

    # Mostrar gráfico
    st.pyplot(plt)

def main():


    st.title('Simulated data for classification task')

    st.write("In this section, data are simulated from bivariate normal distributions "
             "for the explanatory variables x1, x2 and a binary response variable y is also generated. "
             "The sample contains 1000 data and it is a balanced classification task."
             )
    st.write("Below you can see a dataframe with the generated data.")
    st.dataframe(df, width=900)

    st.write("And here, you can see a dispersion graph discriminated by classes.")
    # Graficar datos
    plot_data(X, y)

    # Configurar la aplicación
    st.title('Prediction using Random Forest')

    st.write("In this section a classification model based on random forests is trained, "
             "and below you can enter the values ​​of the explanatory variables x1 and x2 "
             "to generate a prediction of the class.")

    # Crear entradas para las variables
    var1 = st.number_input('Enter the value for variable x1', value=0.0)
    var2 = st.number_input('Enter the value for variable x2', value=0.0)

    if st.button('Make prediction'):
        # Preparar los datos para la predicción
        data = np.array([[var1, var2]])
        
        # Hacer la predicción y obtener las probabilidades
        prediction = model.predict(data)
        prediction_proba = model.predict_proba(data)
        
        # Mostrar el resultado
        st.write(f'The model prediction is: Class {prediction[0]}')
        
        # Mostrar la probabilidad de la clase predicha
        class_probs = prediction_proba[0]
        st.write(f'Probability of Class 0: {class_probs[0]:.2f}')
        st.write(f'Probability of Class 1: {class_probs[1]:.2f}')
    st.write("Now, we will show the decision boundary of the trained model.")
    # Graficar la frontera de decisión
    plot_decision_boundary(model, X, y)

if __name__ == '__main__':
    main()
