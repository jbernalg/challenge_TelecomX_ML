# utils.py
import numpy as np

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



