import numpy as np
import panda as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

file = open("dataset-full.csv", "r")

df = pd.read_csv(file, names=['sepal length','sepal width','petal length','petal width','target'])