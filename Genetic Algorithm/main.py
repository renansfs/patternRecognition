import Algorithm
import pandas as pd

if __name__ == "__main__":
    filename = 'breast_cancer_final.csv'
    breast_cancer_dataset = pd.read_csv(filename)
    b_dict = breast_cancer_dataset.var().to_dict()
    #f = ["A", "B","C","D","E","F"]
    #g = ["E", "D","C","B","A"]
    #d = { "E": 12, "D":25, "C": 75, "B": 30, "A":55}
    #Algorithm.crossover(crossover_rate=100, genome=g, variances=d, features = f)