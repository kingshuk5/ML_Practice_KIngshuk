import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt


#input data set

data=np.array([
    [1,0,1,0],
    [1,0,0,0],
    [0,1,1,0],
])
data1=np.array([
    [1,2,1,0],
    [1,0,0,0],
    [0,1,4,0],
])

#df=np.loadtxt('data.csv',delimiter=',')
#data=pd.DataFrame(df)
#SOM parameters

n_clusters=2 #number of clusters
iterations=10 #number of iterations
alpha_initial=0.5 #initial learning rate
sigma_initial=1.0 #initial neighbourhood radius

#initializing weights
np.random.seed(42)
weights=np.random.rand(n_clusters,data.shape[1])

#Function to decay learning rate and neighbourhood radius
def decay(parameter,initial_value,iteration,total_iterations):
    return initial_value*(1-iteration/total_iterations)

#Training SOM
for t in range(iterations):
    alpha=decay(alpha_initial,alpha_initial,t,iterations)
    sigma=decay(sigma_initial,sigma_initial,t,iterations)

    for x in data:
        #Calculate distance to all neurons
        distances=np.linalg.norm(weights-x,axis=1)
        bmu_idx=np.argmin(distances) #Find the best matching unit

        #update weights for BMU and its neighbours
        for i in range(n_clusters):
            distance_to_bmu=np.abs(i-bmu_idx)
            if distance_to_bmu<=sigma:
                influence=np.exp(-distance_to_bmu**2/(2*sigma**2))
                weights[i] += alpha*influence*(x-weights[i])

#Assign each data point to the closest cluster
cluster_assignments=[np.argmin(np.linalg.norm(weights-x,axis=1)) for x in data1]

#plot raw data and clusters
def plot_clusters(data,clusters_assignments,weights):
    colors=['r','g','b','y']#colors for each cluster
    plt.figure(figsize =(8,6))
    '''
        #plot raw data
        for i,x in enumerate(data):
        plt.scatter(x[0],x[2],c=colors[clusters_assignments[i]],lablel=f"Data point {i}")
    '''
    #plot clusters
    for i,w in enumerate(weights):
        plt.scatter(w[0],w[2],c=colors[i],edgecolors='k',s=200,label=f"Cluster {i+1} Center")

    plt.title("SOM Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid()
    plt.show()

#call the plot  function 
plot_clusters(data1,cluster_assignments,weights)
print("The Clusters of corsponding data points are :")
for i in  range(len(cluster_assignments)):
    print(cluster_assignments[i])


    