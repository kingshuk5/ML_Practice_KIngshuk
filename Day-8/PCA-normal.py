import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#create the dataset
'''
data={

        'X1':[2.5,0.5,2.2,1.9,3.1,2.3,2.0,1.0,1.5,1.1],
        'X2':[2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9]
    }

'''

data=np.loadtxt('Day-5\data.txt',)

df=pd.DataFrame(data)

#standardize the data
mean=df.mean()
std=df.std()
df_standardized=(df-mean)/std

print("standarized data:")
print(df_standardized)

#step3 compute the covariance matrix
cov_matrix=np.cov(df_standardized.T)
print("\nCovariance Matrix:")
print(cov_matrix)

#step 4 compute eignevalues and eignvectors
eigenvalues,eigenvectors=np.linalg.eig(cov_matrix)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

#step5 sort eigenvalues and eigenvectors
sorted_idx=eigenvalues.argsort()[::-1]
sorted_eigenvalues=eigenvalues[sorted_idx]
sorted_eigenvectors=eigenvectors[:,sorted_idx]

print("\nSorted Eigenvalues:")
print(sorted_eigenvalues)
print("\nSorted_eiegnvalues:")
print(sorted_eigenvectors)

#step 6 project the data onto principal components
W=sorted_eigenvectors[:,:2]
projected_data=df_standardized.dot(W)
print("\nProjected data on PCI:")
print(projected_data)

#optionl: visulaise the projection
plot_data=np.array(projected_data)
#define colors and symbols for each row
colors=['red','blue','green','purple','orange','cyan']
symbols=['o','s','','D','P','*']

#plotting
plt.figure(figsize=(8,6))
group1=16
group2=16

data_modified=np.array(df)
#select column for scatter plot
x=data_modified[:,0]
y=data_modified[:,1] 
#plot the scatter plot
plt.scatter(x,y,color=colors[0],marker=symbols[0],label=f'Raw data',s=100)
#labels and legends
plt.xlabel("Column 1")
plt.ylabel("Column 2")
plt.title("Raw data plot")
plt.legend()
plt.grid()
plt.show()

for i in range(group1):
    x,y=plot_data[i]
    plt.scatter(x,y,color=colors[1],marker=symbols[0],label=f'PC {1}' if i==0 else None,s=100)

for i in range(group1,group1+group2):
    x,y=plot_data[i]
    plt.scatter(x,y,color=colors[2],marker=symbols[1],label=f'PC {2}' if i== group1 else None,s=100)

#labels and legends
plt.xlabel("PC 1")
plt.ylabel("Pc 2")
plt.title("PCA plot")
plt.legend()
plt.grid()

#show the plot
plt.show()