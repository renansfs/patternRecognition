#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data


# In[55]:


#Open the file and read it as table
file = open("dataset-full2.csv", "r")

df = pd.read_csv(file, names=['Class', 'Age', 'Age mean', 'Menopause','Tumor-size', 'Tumor-size mean', 'Inv-nodes', 'Inv-nodes mean', 'Node-caps', 'Deg-malig', 'Breast', 'Breast-quad', 'Irradiat'])
print(df)


# In[22]:


#Get just the features below to calculate to transform it to the standard scaler (Normalization)
features = ['Age mean', 'Tumor-size mean', 'Inv-nodes mean']

#Separating out the features
x = df.loc[:, features].values

#Separating out the target
y = df.loc[:, ['Class']]

x = StandardScaler().fit_transform(x)
print (x)


# In[27]:


#Tell PCA how many components it will get
#Get the PCA table

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
print(principalDf)


# In[30]:


#Append the class to the PCA TABLE
finalDf = pd.concat([principalDf, df[['Class']]], axis = 1)
print(finalDf)


# In[52]:


#Create the 3d chart
fig = plt.figure(figsize= (10,10))
ax = fig.add_subplot(111, projection='3d') 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
targets = ['no-recurrence-events', 'recurrence-events']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Class'] == target
    ax.scatter(  finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[46]:


#Variance Values for each principal component analised
pca.explained_variance_ratio_

