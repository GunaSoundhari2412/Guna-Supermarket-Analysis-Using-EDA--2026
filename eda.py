import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# CREATE FOLDER FOR SAVING GRAPHS
# -----------------------------
os.makedirs("graphs", exist_ok=True)

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("supermarket_sales-Sheet1.csv")

print("=" * 60)
print("SUPERMARKET SALES EDA PROJECT")
print("=" * 60)

# -----------------------------
# BASIC INFORMATION
# -----------------------------
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# STATISTICAL SUMMARY
# -----------------------------
print("\nStatistical Summary:")
print(df.describe())

# -----------------------------
# AVERAGE SALES
# -----------------------------
print("\nAverage Sales Sales:")
print(round(df["Sales"].mean(), 2))

# -----------------------------
# SALES REVENUE
# -----------------------------
print("\nSales Revenue:")
print(round(df["Sales"].sum(), 2))

# -----------------------------
# CORRELATION ANALYSIS
# -----------------------------
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(
    "graphs/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# -----------------------------
# SALES DISTRIBUTION
# -----------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["Sales"], bins=20, kde=True)
plt.title("Distribution of Sales Sales")
plt.xlabel("Sales Sales")
plt.ylabel("Frequency")
plt.savefig(
    "graphs/sales_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# -----------------------------
# PAYMENT METHODS
# -----------------------------
plt.figure(figsize=(8, 5))
sns.countplot(x="Payment", data=df)
plt.title("Payment Method Usage")
plt.savefig(
    "graphs/payment_method_usage.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# -----------------------------
# PRODUCT LINE SALES
# -----------------------------
product_sales = (
    df.groupby("Product line")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 5))
product_sales.plot(kind="bar")
plt.title("Sales Sales by Product Line")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    "graphs/product_line_sales.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# -----------------------------
# BRANCH SALES
# -----------------------------
branch_sales = df.groupby("Branch")["Sales"].sum()

plt.figure(figsize=(6, 4))
branch_sales.plot(kind="bar")
plt.title("Branch-wise Sales")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig(
    "graphs/branch_sales.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# -----------------------------
# CUSTOMER TYPE
# -----------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="Customer type", data=df)
plt.title("Customer Type Distribution")
plt.savefig(
    "graphs/customer_type_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# -----------------------------
# GENDER DISTRIBUTION
# -----------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")
plt.savefig(
    "graphs/gender_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# -----------------------------
# CUSTOMER RATING DISTRIBUTION
# -----------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["Rating"], bins=10, kde=True)
plt.title("Customer Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.savefig(
    "graphs/customer_rating_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# -----------------------------
# TOP INSIGHTS
# -----------------------------
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

print("\nHighest Sales Product Line:")
print(product_sales.idxmax())

print("\nHighest Revenue Generated:")
print(round(product_sales.max(), 2))

print("\nMost Used Payment Method:")
print(df["Payment"].mode()[0])

print("\nBest Performing Branch:")
print(branch_sales.idxmax())

print("\nAverage Customer Rating:")
print(round(df["Rating"].mean(), 2))

print("\nSales Revenue:")
print(round(df["Sales"].sum(), 2))

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Graphs saved inside 'graphs' folder.")