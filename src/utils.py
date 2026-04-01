# utils.py
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay, precision_recall_curve
import matplotlib.pyplot as plt

def cv_score_interval(results, nombre: str, score: str = 'Recall'):
    '''
    Funcion que calcula e imprime un intervalo de confianza del 95% para una 
    metrica de evaluacion (por defecto, el Recall) obtenida a partir de resultados 
    de validacion cruzada.
    '''
    
    score_mean = np.round(results['test_score'].mean(), 4)
    score_stdev = np.round(results['test_score'].std(), 4)
    lower_b = np.round(score_mean - 2 * score_stdev, 4)
    upper_b = np.round(min(1, score_mean + 2 * score_stdev), 4)
    
    print(f'Recall promedio de {nombre}: {score_mean}')
    print(f'Desviación estándar del Recall de {nombre}: {score_stdev}')
    print(f'El {score} de {nombre} estará entre [{lower_b:.4f},{upper_b:.4f}] con un 95% de confianza')



def plot_mat_confusion(model,
                  x: pd.core.frame.DataFrame, 
                  y: pd.core.series.Series, 
                  name: str,
                  umbral: float=None):
    '''
     grafica la matriz de confusion de un modelo dado
     x: variables predictoras
     y: variable target
    '''
    
    if umbral == None:
        y_pred = model.predict(x)
    else:
        y_probs = model.predict_proba(x)[:, 1]
        y_pred = (y_probs >= umbral).astype(int) 
        
    confmat = confusion_matrix(y_true=y, y_pred=y_pred)

    confmat_plot = ConfusionMatrixDisplay(confmat, display_labels=['Permanecen', 'Cancelan'])

    fig, ax = plt.subplots(figsize=(10,6))
    confmat_plot.plot(ax=ax, colorbar=False, text_kw={'color': 'black', 'fontweight': 'bold'})

    plt.title(name, fontsize=18, fontweight='bold')
    plt.xlabel('Etiquetas Prediccion', fontsize=16)
    plt.ylabel('Etiquetas Reales ', fontsize=16)
    plt.tick_params(axis='both', labelsize=13)
    
    plt.subplots_adjust(right=0.88)
    plt.tight_layout(pad=3.0)
    plt.show()