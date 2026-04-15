import sys
import time
import math
from pyspark import SparkContext, SparkConf
import pandas as pd

import pandas as pd

def load_data(file):
    data = pd.read_csv(file, header=None)
    
    l = []
    for riga in data.values:

        punti = tuple(riga[:-1])
        etichetta = riga[-1]
        
        l.append((punti, etichetta))
        
    return l

def distanza(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def FairFFT(P, ka, kb):
    # We use this nice strategly
    
    # 1. Inizializzazione
    centers = []
    count_a = 0
    count_b = 0
    n = len(P)
    

    distanza_minima = [float('inf')] * n
    
    # 2. Scegliamo il primo punto come primo centro
    primo_elemento = P[0]
    primo_punto, prima_etichetta = primo_elemento
    centers.append((primo_punto, prima_etichetta))
    
    if prima_etichetta == 'A':
        count_a += 1
    else:
        count_b += 1
        
    # 3. Ciclo per trovare i restanti (ka + kb - 1) centri
    for _ in range(1, ka + kb):
        punto_precedente = centers[-1][0] # Prendiamo solo le coordinate dell'ultimo centro aggiunto
        
        max_dist = -1
        next_index = -1
        
        # Aggiorniamo le distanze minime e cerchiamo il candidato più lontano
        for i in range(n):
            point, label = P[i]
            
            # Calcoliamo la distanza tra il punto i e l'ultimo centro aggiunto
            d = distanza(point, punto_precedente)
            
            # Aggiorniamo la distanza minima per il punto i
            if d < distanza_minima[i]:
                distanza_minima[i] = d
            
            # Verifichiamo se questo punto può essere un candidato (rispetto al budget)
            can_add = False
            if label == 'A' and count_a < ka:
                can_add = True
            elif label == 'B' and count_b < kb:
                can_add = True
                
            # Se può essere aggiunto, vediamo se è il più lontano trovato finora
            if can_add and distanza_minima[i] > max_dist:
                max_dist = distanza_minima[i]
                next_index = i
        
        # Aggiungiamo il candidato migliore ai centri
        if next_index != -1:
            nuovo_punto, nuova_etichetta = P[next_index]
            centers.append((nuovo_punto, nuova_etichetta))
            
            if nuova_etichetta == 'A':
                count_a += 1
            else:
                count_b += 1
                
    return centers


def main():
    start = time.time()
    # Prendi il file dagli argomenti (come chiede il compito)
    file_path = sys.argv[1] 
    ka = int(sys.argv[2])
    kb = int(sys.argv[3])
    # L = int(sys.argv[4]) # Ti servirà per Spark dopo

    data = load_data(file_path)
    
    # PASSA 'data' (la lista), NON 'file_path' (la stringa)
    centri = FairFFT(data, ka, kb)
    
    for punto in centri:
        print(punto)
    stop = time.time()
    print('-'*50)
    print(f"Tempo di esecuzione: {stop-start}")
    print('-'*50)
main()


