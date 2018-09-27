import Algorithm
import pandas as pd

features = ['class','node-caps','deg-malig','breast','irradiat','age_20-29','age_30-39','age_40-49','age_50-59','age_60-69','age_70-79','menopause_ge40','menopause_lt40','menopause_premeno','tumor-size_0-4','tumor-size_10-14','tumor-size_15-19','tumor-size_20-24','tumor-size_25-29','tumor-size_30-34','tumor-size_35-39','tumor-size_40-44','tumor-size_45-49','tumor-size_5-9','tumor-size_50-54','inv-nodes_0-2','inv-nodes_12-14','inv-nodes_15-17','inv-nodes_24-26','inv-nodes_3-5','inv-nodes_6-8','inv-nodes_9-11','breast-quad_central','breast-quad_left_low','breast-quad_left_up','breast-quad_right_low','breast-quad_right_up']

if __name__ == "__main__":
    filename = 'breast_cancer_final.csv'
    breast_cancer_dataset = pd.read_csv(filename, names=features)
    result = Algorithm.genetico(breast_cancer_dataset, max_genoma_size = 5, mutation_rate=0.03, crossover_rate=0.5, max_generation_times=100, features=features, class_name='class', max_gene_size=3)
    value = Algorithm.getBestGeneration(result)
    print(result)