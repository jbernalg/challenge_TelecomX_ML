# Desafio TelecomX (Parte 2)

![Portada Clasificacion](https://github.com/user-attachments/assets/72feb324-bfae-4523-9467-167bf9c649fa)

## Descripcion del Desafio
El proyecto consiste en crear un modelo de clasificación que prediga si un cliente permanece o cancela el servicio con la empresa TelecomX, a partir de características sociodemográficas y uso de servicios. Se busca anticipar el problema de la cancelación para mejorar la retención de clientes. 

## Analisis del problema
- Es un problema de clasificacion binaria.
- La variable objetivo es 'cancelacion'. El cliente se irá o no?
- Los tipos de modelo a usar será de clasificación. Probamos dos modelos: DecisionTree y RandomForest.

## Tareas 
✔ Preparar los datos para el modelo.
✔ Realizar análisis de correlación.
✔ Entrenar los modelos de clasificación.
✔ Evaluar el rendimiento de los modelos con métricas.
✔ Seleccionar mejor modelo.

## Evaluacióm de modelos
Los modelos evaluados fueron:

- RandomForest
- DecisionTree
  
Se eliminan las columnas irrelevantes (CustomID, meses_contrato, cuenta mensual). Utilizamos la codificacion categorica con OneHotEncoder. Al final se utiliza el balanceo de clases 'class_weight' para RandomForest y validación cruzada.

## Modelos Seleccionados
En función de las métricas obtenidas en los entrenamientos de los modelos, elegimos como el mejor al RandomForest. Los parametros 
    'n_estimators': 200
    'max_depth': 10
    'min_samples_split': 2
    'min_samples_leaf': 4
    'max_features': 'log2'
    'class_weight': {0: 1, 1: 3}
    'random_state': 42

## Tecnologías usadas
- Python
- Pandas y Numpy.
- ScikitLearn, Statsmodels, ImbLean, Scipy
- Matplotlib y Seaborn.

## Autor

[<img src="https://avatars.githubusercontent.com/u/99054174?v=4" width=115><br><sub>Jeinfferson Bernal</sub>](https://github.com/jbernalg)
