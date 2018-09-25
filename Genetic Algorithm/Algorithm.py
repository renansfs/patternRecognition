import pandas as pd
import numpy as np

def genetico(dataFrame, n_generation, mutation_rate):
    pass

def mutacao(df: pd.DataFrame, mut_rate, genome, features):
    prob = np.random.randint(100)
    if mut_rate > prob:
        indexG = np.random.randint(len(genome))
        indexF = np.random.randint(len(features))
        genome[indexG] = features[indexF]

def crossover(crossover_rate, genome, variances, features):
    #Taxa randomica do crossover
    random = np.random.rand() * np.random.randint(100) 
    cross = []
    for feature in variances:
        if random < variances[feature]:
            index = int(np.random.randint(len(features)))
            cross.append(features[index])
        else:
            cross.append(feature)
    return cross
    
def variances(df: pd.DataFrame):
    variance = pd.DataFrame.var().to_dict()
    return variance

def fitness(df: pd.DataFrame, class_name, target_feature_number, function, best_measurement, maximize_measurement=True):
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
    return distance

def euclidian_distance(arr1, arr2):
    if len(arr1) != len(arr2):
        raise Exception('Euclidian distance: arrays of differente lenghts!')

    distance = 0
    for i in range(0, len(arr1)):
        distance += (arr1[i] - arr2[i])**2
    return math.sqrt(distance)