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
def genetico(df: pd.DataFrame, max_genoma_size, mutation_rate, crossover_rate, max_generation_times, features, class_name, max_gene_size):
    generations = dict()
    generation = list()
    bestGenome = list()
    
    #Cria o varios genomas por geracao
    for _ in range(max_gene_size):
        genome = list()
        for _ in range(np.random.randint(len(features))):
            index = features[np.random.randint(len(features))]
            genome.append(index)
        generation.append(genome)

        genome = generation[0]
        length = 0
        #Maxima quantidade de rodadas por geracao
        for _ in range(max_generation_times):
            #Seleciona apenas as dimensoes do genome atual
            evaluate = df.loc[:, genome]
            #Retorna o dicionario com variancias e dimensoes
            variance = variances(evaluate)
            cross = crossover(crossover_rate, genome, variance, generation)
            mutacao(df, mutation_rate, cross, features)
            #Concatena a classe a esse genoma
            gen = pd.concat([evaluate, df[[class_name]]], axis=1)
            #Verifica o resultado com a distancia
            length = fitness(gen, class_name=class_name, target_feature_number=max_genoma_size)
            #Reinsere o genoma
            genome = cross
        generations[length] = genome  
    return generations

#Mutacao
#df: Datafrma
#mut_rate: taxa da mutacao maxima
#genoma: features do genoma
#features: features total
def mutacao(df: pd.DataFrame, mut_rate, genome, features):
    #Random ate 100
    prob = np.random.randint(100)
    #Verificar se a mutacao vai acontecer
    if mut_rate > prob:
        #Pega o index do set total e adiciona no indexG random
        indexG = np.random.randint(0, len(genome))
        indexF = np.random.randint(0, len(features))
        genome[indexG] = features[indexF]


#Taxa randomica do crossover
#Crossover_rate: taxa maxima para acontecer o crossover
#genome: features do genoma
#variances: lista com variancias
#features: features total
def crossover(crossover_rate, genome, variances, generation):
    #Random para deletar as dimensoes conforme o crossover_Rate
    random = np.random.rand() * crossover_rate
    #Nome do novo genoma
    newcross = []
    #Como o genoma == variancias
    for feature in variances:
        #se o random for menor que variancia da dimensao atual
        if random < variances[feature]:
            #Index random da geracao
            indexGeneration = int(np.random.randint(len(generation)))
            #Index random do Gene
            indexGene = int(np.random.randint(len(generation[indexGeneration])))
            #Adiciona a nova geracao aquele gene
            newcross.append(generation[indexGeneration][indexGene])
        else:
            #Caso seja menor, adiciona a atual
            newcross.append(feature)
        #Retorna o genoma novo (filho)
    return newcross

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