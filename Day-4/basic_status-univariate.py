import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew,kurtosis

#Generate or  load data (you can replace this with your data)
data={
    "A":np.random.normal(50,10,5),
    "B":np.random.uniform(20,80,5),
    "C":np.random.poisson(30,5)
}
print(data)
df=pd.DataFrame(data)
df.to_csv("raw_data.csv",index=False)
#Compute statistics
statistics=[]
for column in df.columns:
    col_data=df[column]
    stats={
        "Column":column,
        "Mean":col_data.mean(),
        "Median":col_data.median(),
        "Mode":col_data.mode()[0] if not col_data.mode().empty else np.nan,
        "variance": col_data.var(),
        "Standard Deviation":col_data.std(),
        "Skewness": skew(col_data),
        "Krutosis": kurtosis(col_data),
        "Min":col_data.min(),
        "Max":col_data.max(),
        "Range":col_data.max()-col_data.min(),
    }
    statistics.append(stats)

#Save statistic to csv
stats_df=pd.DataFrame(statistics)
#print(statistics)
stats_df.to_csv("Statistics_summary.csv",index=False)
print("statistics saved to 'Statistics_summary.cs'")

#visulalization
for column in df.columns:
    plt.figure(figsize=(15,5))

    #Box Plot
    plt.subplot(1,3,1)
    sns.boxplot(y=df[column])
    plt.title(f"Box Plot:{column}")

    #Create the line plot
    plt.subplot(1,3,1)
    plt.plot(df[column],marker='o',linestyle='-',color='b',label='Line 1')
    plt.title(f"Line Graph: {column}")

    #Histogram
    plt.subplot(1,3,2)
    plt.hist(df[column],bins=20,edgecolor='black',alpha=0.7)
    plt.title(f"Histogram:{column}")

    #Bar Plot (frequency Counts)
    plt.subplot(1,3,3)
    value_counts=df[column].value_counts().head()
    sns.barplot(x=value_counts.index,y=value_counts.values)
    plt.title(f"Bar plot:{column}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#correction matrix
plt.figure(figsize=(10,8))
corr_matrix=df.corr()
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Correaltion Coefficient Matrix")
plt.show()
