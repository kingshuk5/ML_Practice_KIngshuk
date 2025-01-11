import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

#load the data set
data= np.loadtxt('Day-5\data.txt')
#data= np.genformtxt('data.txt),

df=pd.DataFrame(data)
#Number of clusters
k=3

#Initialize sentroids randomly
np.random.seed(200)#for reproducibility
initial_idices=np.random.choice(len(data),k,replace=False)
centroids=data[initial_idices]

#functions to calculate euclidean distance
def calculate_distance(point,cetroids):
    return np.linalg.norm(point-centroids,axis=1)

#Interate to assign clusters and update centroids
max_iterations=10``
for iteration in range(max_iterations):
    #Assign clusters
    cluster_labels=np.array([
        np.argmin(calculate_distance(point,centroids)) for point in data
    ])

    #calculate new centroids
    new_centroids=np.array([
        data[cluster_labels==i].mean(axis=0) for i in range(k)
    ])

    #check for convergence
    if np.all(centroids==new_centroids):
        break
    centroids=new_centroids

#visualize the results
for i in range(k):
    cluster_points=data[cluster_labels==i]
    plt.scatter(cluster_points[:,0],cluster_points[:,1],label=f'Cluster {i+1}')
plt.scatter(centroids[:,0],centroids[:,1],color='black',marker='x',label='Centroids')
plt.legend()
plt.title('K-means clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()