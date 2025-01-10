import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate or load data
data = {
    "A": np.random.normal(50, 10, 5),
    "B": np.random.uniform(20, 80, 5),
    "C": np.random.poisson(30, 5),
    "D": np.random.uniform(20, 100, 5),
    "E": np.random.poisson(30, 5)
}
print(data)
df = pd.DataFrame(data)
df.to_csv("raw_data.csv", index=False)

# User-defined statistics calculations
def calculate_statistics(data):
    """Calculate statistics manually for a list of values."""
    n = len(data)
    mean = sum(data) / n
    sorted_data = sorted(data)
    median = (
        sorted_data[n // 2]
        if n % 2 == 1
        else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    )
    mode = max(set(data), key=data.count) if len(set(data)) < n else np.nan
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = variance ** 0.5
    minimum = min(data)
    maximum = max(data)
    data_range = maximum - minimum
    return {
        "Mean": mean,
        "Median": median,
        "Mode": mode,
        "Variance": variance,
        "Standard Deviation": std_dev,
        "Min": minimum,
        "Max": maximum,
        "Range": data_range
    }

# Compute statistics
statistics = []
for column in df.columns:
    stats = calculate_statistics(df[column].tolist())
    stats["Column"] = column
    statistics.append(stats)

# Save statistics to CSV
stats_df = pd.DataFrame(statistics)
stats_df.to_csv("Statistics_summary.csv", index=False)
print("Statistics saved to 'Statistics_summary.csv'")

# Visualization
for column in df.columns:
    plt.figure(figsize=(20, 6))

    # Box Plot
    plt.subplot(1, 5, 1)
    sns.boxplot(y=df[column])
    plt.title(f"Box Plot: {column}")

    # Line Plot
    plt.subplot(1, 5, 2)
    plt.plot(df[column], marker="o", linestyle="-", color="b", label=column)
    plt.title(f"Line Graph: {column}")
    plt.legend()

    # Histogram
    plt.subplot(1, 5, 3)
    plt.hist(df[column], bins=10, edgecolor="black", alpha=0.7)
    plt.title(f"Histogram: {column}")

    # Bar Plot (Frequency Counts)
    plt.subplot(1, 5, 4)
    value_counts = df[column].value_counts().head()
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title(f"Bar Plot: {column}")
    plt.xticks(rotation=45)

    # Pie Chart
    plt.subplot(1, 5, 5)
    plt.pie(
        df[column],
        labels=[f"Value {i+1}" for i in range(len(df[column]))],
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title(f"Pie Chart: {column}")

    plt.tight_layout()
    plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Coefficient Matrix")
plt.show()
