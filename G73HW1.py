import sys
import time
import math
from pyspark import SparkContext, SparkConf
import pandas as pd



def load_data(file):
    data = pd.read_csv(file, header=None)
    
    l = []
    for riga in data.values:

        punti = tuple(riga[:-1])
        etichetta = riga[-1]
        
        l.append((punti, etichetta))
        
    return l
def parse_point(line):
    """
    Trasforma una riga del CSV in una struttura dati pronta all'uso.
    Input: "1.5,6.0,2.3,A" -> Output: ((1.5, 6.0, 2.3), "A")
    """
    parts = line.split(',')
    # Le coordinate sono tutti i valori tranne l'ultimo
    point = tuple(map(float, parts[:-1]))
    # L'ultimo carattere è la label del gruppo (A o B)
    label = parts[-1].strip()
    return (point, label)
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

def MRFairFFT(inputPoints, ka, kb):
    """
    Implementazione MapReduce in 2 Round.
    Usa FairFFT in parallelo sulle partizioni e poi una sintesi finale.
    """
    # ROUND 1: Calcolo dei centri locali (Coreset) su ogni partizione
    def round1_map(iterator):
        # Ogni partizione viene trasformata in lista e passata a FairFFT
        partition_data = list(iterator)
        if not partition_data: return []
        return FairFFT(partition_data, ka, kb)

    # Coreset_rdd conterrà (L * k) centri distribuiti
    coreset_rdd = inputPoints.mapPartitions(round1_map)

    # ROUND 2: Raccolta dei centri locali sul Driver e calcolo finale
    # collect() è ammesso qui perché il coreset è molto piccolo
    local_coreset = coreset_rdd.collect()
    
    # Esecuzione finale per ottenere esattamente ka e kb centri globali
    return FairFFT(local_coreset, ka, kb)

def main_offline():
    start = time.time()
    # Prendi il file dagli argomenti (come chiede il compito)
    file_path = sys.argv[1] 
    ka = int(sys.argv[2])
    kb = int(sys.argv[3])
    # L = int(sys.argv[4]) # Ti servirà per Spark dopo

    data = load_data(file_path)
    
    # offline
    centri = FairFFT(data, ka, kb)
    
    for punto in centri:
        print(punto)
    stop = time.time()
    print('-'*50)
    print(f"Tempo di esecuzione: {stop-start}")
    print('-'*50)

def main():
    # Lettura parametri da riga di comando
    if len(sys.argv) < 5:
        print("Usage: python GxxHW1.py <file_path> <ka> <kb> <L>")
        sys.exit(1)

    file_path = sys.argv[1]
    ka = int(sys.argv[2])
    kb = int(sys.argv[3])
    L = int(sys.argv[4])

    # Configurazione Spark (Context)
    conf = SparkConf().setAppName(f"GxxHW1")
    sc = SparkContext(conf=conf)

    # 1. Stampa parametri iniziali
    print(f"INPUT PARAMETERS: file={file_path} ka={ka} kb={kb} L={L}")

    # 2. Lettura dati in RDD distribuito (OBBLIGATORIO)
    # .cache() serve perché useremo l'RDD più volte (count e algoritmo)
    inputPoints = sc.textFile(file_path, minPartitions=L).map(parse_point).cache()

    # 3. Statistiche NA e NB
    N = inputPoints.count()
    NA = inputPoints.filter(lambda x: x[1] == 'A').count()
    NB = inputPoints.filter(lambda x: x[1] == 'B').count()
    print(f"N={N}, NA={NA}, NB={NB}")

    # 4. Esecuzione MRFairFFT con misurazione tempo
    start_t = time.time()
    solution = MRFairFFT(inputPoints, ka, kb)
    end_t = time.time()
    elapsed_ms = (end_t - start_t)

    # 5. Calcolo Objective Function: max_{x in U} dist(x, S)
    # Lo facciamo con Spark per scalare su dataset enormi
    centers_coords = [c[0] for c in solution]
    
    def get_min_dist(p_tuple):
        p = p_tuple[0]
        return min(distanza(p, c) for c in centers_coords)

    objective_value = inputPoints.map(get_min_dist).max()

    # 6. OUTPUT FINALE (Formato richiesto)
    for c_point, c_label in solution:
        print(f"Center: {c_point} Label: {c_label}")
    
    print(f"Max distance = {objective_value}")
    print(f"Time of MRFairFFT = {int(elapsed_ms)} ms")

