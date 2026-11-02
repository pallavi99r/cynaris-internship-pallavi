import numpy as np

# 1D Array
array_1D = np.array([10, 20, 30, 40])
print("1D Array:", array_1D)
print("1D Shape:", array_1D.shape)

# 2D Array
array_2D = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D Array:\n", array_2D)
print("2D Shape:", array_2D.shape)

# 3D Array
array_3D = np.array(
    [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ]
)
print("\n3D Array:\n", array_3D)
print("3D Shape:", array_3D.shape)

# Broadcasting
numbers = np.array([10, 20, 30])
result = numbers + 5

print("\nBroadcasting:", result)

# Vectorised operations
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print("\nAddition:", a + b)
print("Substraction:", a - b)
print("Multiplication:", a * b)
print("Division:", b / a)
print("Square:", a**2)

# Matrix multiplication
matrix_a = np.array([[1, 2], [3, 4]])

matrix_b = np.array([[5, 6], [7, 8]])

result = np.matmul(matrix_a, matrix_b)

print("\nMatrix Multiplication:\n", result)


# Statistics using the Iris CSV dataset

from pathlib import Path

# Get the project folder
project_folder = Path(__file__).resolve().parent.parent

# Load numeric columns from CSV
data = np.loadtxt(
    project_folder / "data" / "iris.csv",
    delimiter=",",
    skiprows=1,
    usecols=(0, 1, 2, 3),
)

# Calculate mean
mean_values = np.mean(data, axis=0)

# Calculate standard deviation
std_values = np.std(data, axis=0)

# Calculate correlation
correlation_matrix = np.corrcoef(data, rowvar=False)

print("\nMean:")
print(mean_values)

print("\nStandard Deviation:")
print(std_values)

print("\nCorrelation Matrix:")
print(correlation_matrix)
