# assignment_dataset.py
"""
Data Analysis & Visualization Project
- Loads the Iris dataset
- Explores and cleans the data
- Performs simple analysis
- Creates four visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ---------------- Task 1: Load & Explore Dataset ----------------
try:
    # Load iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame

    print("✅ First 5 rows of dataset:")
    print(df.head(), "\n")

    print("✅ Dataset info:")
    print(df.info(), "\n")

    print("✅ Missing values per column:")
    print(df.isnull().sum(), "\n")

    # Clean (drop missing if any)
    df = df.dropna()

except FileNotFoundError:
    print("❌ Dataset file not found.")
    exit()
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# ---------------- Task 2: Basic Analysis ----------------
print("📊 Basic Statistics:")
print(df.describe(), "\n")

print("📊 Group Means by Species:")
group_means = df.groupby("target").mean()
print(group_means, "\n")

# ---------------- Task 3: Visualizations ----------------
plt.style.use("seaborn-v0_8")

# 1. Line chart (trend for first 40 samples)
plt.figure()
plt.plot(df.index[:40], df["sepal length (cm)"][:40], marker="o", color="blue", label="Sepal Length")
plt.title("Sepal Length Trend (First 40 Samples)")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar chart (average petal length per species)
plt.figure()
group_means["petal length (cm)"].plot(kind="bar", color="skyblue")
plt.title("Average Petal Length per Species")
plt.xlabel("Species (0=setosa, 1=versicolor, 2=virginica)")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram (sepal width distribution)
plt.figure()
plt.hist(df["sepal width (cm)"], bins=15, color="green", edgecolor="black")
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (sepal length vs petal length)
plt.figure()
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"], c=df["target"], cmap="viridis")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.colorbar(label="Species")
plt.show()
