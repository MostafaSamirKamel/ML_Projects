# ML_Projects

---

# **Customer Segmentation Project Documentation**

## **1. Project Steps Overview**

The project aims to perform **customer segmentation** based on their purchasing behavior using unsupervised learning techniques. The steps involved are:

1. **Data Loading & Exploration**:
   - Load the dataset and understand its structure.
   - Generate statistical summaries to check for any data issues.
   
2. **Data Preprocessing**:
   - Encode categorical variables (like Gender).
   - Handle missing data, if any.
   - Analyze and visualize distributions of numerical features.
   - Detect and handle outliers.

3. **Dimensionality Reduction**:
   - Apply **Principal Component Analysis (PCA)** to reduce high-dimensional data into fewer components while preserving information.

4. **Clustering**:
   - Use **K-means clustering** to group customers into segments based on their attributes.
   - Visualize clusters in both 2D and 3D.

5. **Comparison with DBSCAN**:
   - Apply **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) and compare it with K-means.
   - Evaluate clustering results using **Silhouette scores**.

6. **Results Evaluation**:
   - Visualize and interpret the clustering results from both K-means and DBSCAN.
   - Compare the clustering quality and understand which algorithm is more suited for the problem.

---

## **2. Code Explanation & Results**

### 1. **Loading the Dataset**
```python
file_path = 'Mall_Customers.csv' 
data = pd.read_csv(file_path)
```
- **Purpose**: Loads the CSV dataset into a Pandas DataFrame. `data` now holds the entire dataset, and you can manipulate and analyze it using Pandas functions.

---

### 2. **Dataset Overview**
```python
def dataset_overview(data):
    print("Dataset Overview:")
    print(f"Number of Records: {data.shape[0]}")
    print(f"Number of Columns: {data.shape[1]}")
    print(data.info())
```
- **Purpose**: Prints a basic overview of the dataset.
  - `data.shape[0]` gives the number of records (rows).
  - `data.shape[1]` gives the number of columns.
  - `data.info()` displays details about the columns, their data types, and how many non-null values exist in each column.
- **Example Output**:
```
Dataset Overview:
Number of Records: 200
Number of Columns: 5
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200 entries, 0 to 199
Data columns (total 5 columns):
  #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
  0   CustomerID              200 non-null    int64 
  1   Gender                  200 non-null    object
  2   Age                     200 non-null    int64 
  3   Annual Income (k$)      200 non-null    int64 
  4   Spending Score (1-100)  200 non-null    int64 
dtypes: int64(4), object(1)
memory usage: 7.9+ KB
None
  
```
### 3. **Statistical Summary**
```python
def statistical_summary(data):
    print("\nStatistical Summary:")
    print(data.describe())
```
- **Purpose**: Displays the statistical summary for numerical columns in the dataset.
  - `.describe()` provides:
    - Count, mean, std (standard deviation), min, max, and various percentiles (25%, 50%, 75%) for numerical columns.
- **Example Output**:
```
Statistical Summary:
        CustomerID         Age  Annual Income (k$)  Spending Score (1-100)
count  200.000000  200.000000          200.000000              200.000000
mean   100.500000   38.850000           60.560000               50.200000
std     57.879185   13.969007           26.264721               25.823522
min      1.000000   18.000000           15.000000                1.000000
25%     50.750000   28.750000           41.500000               34.750000
50%    100.500000   36.000000           61.500000               50.000000
75%    150.250000   49.000000           78.000000               73.000000
max    200.000000   70.000000          137.000000               99.000000
```
---

### 4. **Label Encoding for Categorical Data**
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
```
- **Purpose**: Converts categorical values into numerical values.
  - **`LabelEncoder`** is used to encode the 'Gender' column, where 'Male' is encoded as 1 and 'Female' as 0.
  - **`fit_transform`** learns the mapping and applies it.

---

### 5. **Distribution Analysis**
```python
def distribution_analysis(data, numerical_columns):
    distribution_analysis = data[numerical_columns].agg(['mean', 'std', 'min', 'max', 'skew', 'kurt'])
    print("\nDistribution Analysis:")
    print(distribution_analysis)
```
- **Purpose**: Analyzes the distribution of numerical columns.
  - **`.agg()`** is used to calculate multiple summary statistics (mean, standard deviation, min, max, skewness, kurtosis) on the selected numerical columns.
  - **Skew**: Measures the asymmetry of the distribution.
  - **Kurt**: Measures the "tailedness" or peakedness of the distribution.
- **Example Output**:
```
Distribution Analysis:
      Gender       Age         Annual Income (k$)  Spending Score (1-100)
mean  0.440000  38.850000           60.560000               50.200000
std   0.497633  13.969007           26.264721               25.823522
min   0.000000  18.000000           15.000000                1.000000
max   1.000000  70.000000          137.000000               99.000000
skew  0.243578   0.485569            0.321843               -0.047220
kurt -1.960375  -0.671573           -0.098487               -0.826629

```
```python
def visualize_distributions(data, numerical_columns):
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(numerical_columns):
        plt.subplot(1, 4, i + 1)
        sns.histplot(data[col], kde=True, bins=15, color='skyblue')
        plt.title(f"{col} Distribution")
    plt.tight_layout()
    plt.show()
```
- **Purpose**: Plots the distribution of each numerical column.
  - **`sns.histplot()`**: Creates a histogram with KDE (Kernel Density Estimation) overlaid to visualize the distribution of values.

![Distribution Visualization](C:\Users\Mostafa Samir\Desktop\ML_Project\dis.png)
---

### 6. **Correlation Analysis**
```python
def correlation_analysis(data, numerical_columns):
    correlation_matrix = data[numerical_columns].corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
```
- **Purpose**: Computes the correlation matrix to identify relationships between numerical variables.
  - **`.corr()`** calculates the Pearson correlation coefficient between pairs of numerical columns. The value ranges from -1 (negative correlation) to 1 (positive correlation).
  
```python
def plot_pairplot(data, numerical_columns):
    sns.pairplot(data[numerical_columns])
    plt.suptitle("Pairplot of Numerical Variables", y=1.02)
    plt.show()

```

---

## **3. K-means Algorithm Explanation**

**K-means clustering** is a partition-based algorithm that aims to divide data into **K clusters** by minimizing the variance within each cluster. It works as follows:

1. **Initialization**: Choose K initial centroids (randomly or using methods like K-means++). 
2. **Assignment Step**: Assign each data point to the nearest centroid.
3. **Update Step**: Calculate the new centroids by taking the mean of all points assigned to each centroid.
4. **Repeat**: Steps 2 and 3 are repeated until convergence (when centroids no longer change).

### **Advantages**:
- Efficient and widely used.
- Works well when clusters are spherical and well-separated.

---

## **4. DBSCAN Algorithm Explanation**

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) works by finding regions of high density in the data:

1. **Core Points**: Points with more than a specified number of neighbors within a given radius (ε).
2. **Border Points**: Points that have fewer neighbors than the core point but are within ε distance from a core point.
3. **Noise**: Points that are neither core nor border points.

DBSCAN does not require the number of clusters to be specified and can find arbitrarily shaped clusters.

### **Advantages**:
- Can identify noise (outliers).
- Works well with clusters of non-spherical shapes.

---

## **5. Comparison Between K-means and DBSCAN**

| **Feature**               | **K-means**                         | **DBSCAN**                          |
|---------------------------|-------------------------------------|-------------------------------------|
| **Cluster Shape**          | Spherical, well-separated          | Arbitrary shapes                    |
| **Number of Clusters**     | Must be specified                  | Automatically determined             |
| **Sensitivity to Noise**   | Cannot handle noise (outliers)     | Can identify noise                   |
| **Scalability**            | Scales well with large datasets    | May struggle with large datasets    |
| **Cluster Density**        | Assumes clusters have equal density| Can handle clusters with varying density |

---

## **6. Why K-means is Better than DBSCAN**

1. **Cluster Shape**: K-means assumes clusters are spherical, making it more suitable for problems with such structures. If your data tends to have well-separated, spherical clusters, K-means will likely perform better.
   
2. **Scalability**: K-means is more scalable for large datasets because its time complexity is generally lower compared to DBSCAN.

3. **Noise Handling**: While DBSCAN excels at identifying noise, K-means can still perform well if the dataset is clean and noise-free. For customer segmentation, noise may be minimal, making K-means more practical.

4. **Performance**: K-means often performs faster, especially when using optimization techniques like K-means++ to initialize centroids.

---

### **7. Visualizations**

1. **2D Scatter Plot**:

   Visualizes how the clusters are separated in the PCA-transformed data.

   **Example Visualization**:
   ![2D Scatter](path_to_image)

2. **3D Scatter Plot**:

   Provides a more detailed view of the clusters in 3-dimensional space.

   **Example Visualization**:
   ![3D Scatter](path_to_image)

---

### **Conclusion**

In this project, we successfully implemented customer segmentation using K-means and DBSCAN. By evaluating both methods, we concluded that K-means is generally more effective for well-separated, spherical clusters, which is often the case in customer segmentation tasks. However, DBSCAN remains a good choice if we expect arbitrary-shaped clusters and need to identify noise.
