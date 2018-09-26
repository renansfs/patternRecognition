import pandas as pd
import numpy as np
import math as mt
import operator as op

def getBestGeneration(generations):
    return max(generations.items(), key=op.itemgetter(1))[0]


#Chamada do Algoritmo
#df: dataframe
#n_gene: maxima numero de geracoes
#mutation_rate: taxa de mutacao
#crossover_rate: taxa de crossover
#features name
#Retonar a melhor geracao
def genetico(df: pd.DataFrame, max_genoma_size, mutation_rate, crossover_rate, max_generation_times, features, class_name):
    generations = dict()
    genome = list()
    
    for _ in range(max_genoma_size):
        index = features[np.random.randint(len(features))]
        genome.append(index)

    for _ in range(max_generation_times):
        evaluate = df.loc[:, genome]
        variance = variances(evaluate)
        cross = crossover(crossover_rate, genome, variance, features)
        mutacao(df, mutation_rate, cross, features)
        generation = pd.concat([evaluate, df[[class_name]]], axis=1)
        measure = fitness(generation, class_name=class_name, target_feature_number=max_genoma_size)
        generations[measure] = cross
    return generations

#Mutacao
#df: Datafrma
#mut_rate: taxa da mutacao maxima
#genoma: features do genoma
#features: features total
def mutacao(df: pd.DataFrame, mut_rate, genome, features):
    prob = np.random.randint(100)
    if mut_rate > prob:
        indexG = np.random.randint(len(genome))
        indexF = np.random.randint(len(features))
        genome[indexG] = features[indexF]


#Taxa randomica do crossover
#Crossover_rate: taxa maxima para acontecer o crossover
#genome: features do genoma
#variances: lista com variancias
#features: features total
def crossover(crossover_rate, genome, variances, features):
    random = np.random.rand() * crossover_rate
    cross = []
    for feature in variances:
        if random < variances[feature]:
            index = int(np.random.randint(len(features)))
            cross.append(features[index])
        else:
            cross.append(feature)
    return cross

#Variancias
#df: DataFrame
#retorna o dicionario de variancias
def variances(df: pd.DataFrame):
    variance = df.var().to_dict()
    return variance

def fitness(df: pd.DataFrame, class_name, target_feature_number):
    try:
        # separa os grupos de classe 0 e 1
        group_0 = df[df[class_name] == 0]
        group_1 = df[df[class_name] == 1]

        # retira os campos classe para nao computar distancia entre eles
        group_0 = group_0.drop(class_name, axis=1)
        group_1 = group_1.drop(class_name, axis=1)

        # centroide as medias de cada feature
        centroid_0 = []
        for i in group_0:
            centroid_0.append(group_0[i].mean())
        centroid_1 = []
        for i in group_1:
            centroid_1.append(group_1[i].mean())

        distance = euclidian_distance(centroid_0, centroid_1)
    except:
        distance = 0
    return distance

def euclidian_distance(arr1, arr2):
    if len(arr1) != len(arr2):
        raise Exception('Euclidian distance: arrays of differente lenghts!')

    distance = 0
    for i in range(0, len(arr1)):
        distance += (arr1[i] - arr2[i])**2
    return mt.sqrt(distance)