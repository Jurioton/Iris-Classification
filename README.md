# 🌸 Iris Flower Classification

This project classifies iris flowers into three categories — *Setosa*, *Versicolor*, and *Virginica* — using a machine learning technique called **K-Nearest Neighbors (KNN)**. The implementation is done in Python using libraries such as `scikit-learn`, `pandas`, and `matplotlib` in a Google Colab notebook.

---

## 📂 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- **Dataset Size**: 150 rows, 5 columns
- **Features**:
  - `sepal_length` (cm)
  - `sepal_width` (cm)
  - `petal_length` (cm)
  - `petal_width` (cm)
- **Target**:
  - `Class`: Species of the flower — Iris-setosa, Iris-versicolor, Iris-virginica

---

## 📊 Exploratory Data Analysis (EDA)

- Used `pairplot` from `seaborn` to visualize how the classes are distributed based on features.
- Statistical summaries (`mean`, `std`, etc.) were generated using `describe()`.

---

## 🧹 Data Preprocessing

- Loaded the dataset using `pandas`.
- Named the columns explicitly.
- Split the dataset into:
  - **Features (X)** — the first 4 columns.
  - **Target (y)** — the species of the iris.
- Used `train_test_split` to divide the data:
  - **Training Set**: 70%
  - **Test Set**: 30%

---

## 🤖 Algorithm Explained: K-Nearest Neighbors (KNN)

**K-Nearest Neighbors (KNN)** is a supervised learning algorithm used for **classification** and **regression**. In this project, it is used for classification.

### 🔍 Key Idea

KNN doesn't create an explicit model or make assumptions about the data distribution. Instead, it stores the entire training dataset. When asked to classify a new data point, it looks at the 'k' closest training examples and chooses the most frequent label among them.

---

### 📌 Steps in KNN Classification

1. **Select K**:  
   Choose the number of neighbors (k). In our case, `k = 3`.

2. **Measure Distance**:  
   For a new test sample, compute the distance (typically **Euclidean**) to every point in the training set:

   \[
   \text{distance}(x, x_i) = \sqrt{(x_1 - x_{i1})^2 + (x_2 - x_{i2})^2 + \dots + (x_n - x_{in})^2}
   \]

3. **Find Neighbors**:  
   Sort the distances and pick the `k` nearest neighbors (smallest distances).

4. **Vote for the Class**:  
   Check which class appears most frequently among the `k` neighbors, and assign that class to the new data point.

5. **Predict**:  
   Return the predicted label.

---

### 📈 Example:

Imagine a new flower with the following measurements:
- `sepal_length`: 5.1
- `sepal_width`: 3.5
- `petal_length`: 1.4
- `petal_width`: 0.2

KNN will:
- Measure distance from this point to every point in training data.
- Pick the 3 closest neighbors.
- Check which class is most common among those 3.
- Predict that class (likely "Setosa" in this case).

---

### 🧠 Why It Works Well Here

The Iris dataset has **well-separated** classes in the feature space, especially when plotting petal length and width. This makes KNN a good candidate, as nearby points truly do belong to the same class.

---

## 🧪 Model Training and Evaluation

- Trained the model using:

  ```python
  knn = KNeighborsClassifier(n_neighbors=3)
  knn.fit(X_train, y_train)

