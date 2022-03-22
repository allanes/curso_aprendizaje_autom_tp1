from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def preProcesamiento(dataset, categorias):
    # Pre-procesamiento de datos
    for categoria in categorias:
        valores_unicos_array = dataset[categoria].unique()
        # Reordeno en forma de diccionario para mapearlos
        mapa_valores_unicos = {valores_unicos_array[index]: index for index in range(len(valores_unicos_array))}
        print(f'Mapeo de {categoria}: {mapa_valores_unicos}')
        dataset[categoria] = dataset[categoria].map(mapa_valores_unicos)
    
    return dataset

def main():
    dataset = pd.read_csv('./data/data.csv')
    # print(dataset)
    categorias_entrenamiento = ['pelo', 'estatura', 'peso', 'protector']
    categorias_objetivo = ['quemado']
    categorias = categorias_entrenamiento + categorias_objetivo
    
    dataset_procesado = preProcesamiento(dataset, categorias)
    # print(dataset_procesado)
    
    # Splitting del dataset
    input_train, input_test, output_train, output_test = train_test_split(
        dataset_procesado[categorias_entrenamiento], 
        dataset_procesado[categorias_objetivo],
        test_size=0.3
        )
    
    # Entrenamiento del modelo
    arbol_entropia = tree.DecisionTreeClassifier(criterion='entropy')
    arbol_gini = tree.DecisionTreeClassifier(criterion='gini')
    arbol = arbol_entropia
    arbol.fit(input_train, output_train)
    
    # Analizar modelo
    predicciones = arbol.predict(X=input_test)
    matriz_confusion = confusion_matrix(y_true=output_test, y_pred=predicciones)
    widget_matriz_confusion = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion, display_labels=arbol.classes_)
    widget_matriz_confusion.plot()
    plt.show()
    
    # Visualizar arbol
    
    

if (__name__ == '__main__'):
    main()
    