"""NumPy fundamentals: arrays, operations, and Iris dataset statistics."""

from pathlib import Path

import numpy as np


def demonstrate_arrays() -> None:
    """Create 1D, 2D, and 3D arrays and display their shapes."""
    array_1d = np.array([10, 20, 30, 40])
    array_2d = np.array([[1, 2, 3], [4, 5, 6]])
    array_3d = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )

    print(f"1D Array: {array_1d}")
    print(f"1D Shape: {array_1d.shape}")

    print(f"\n2D Array:\n{array_2d}")
    print(f"2D Shape: {array_2d.shape}")

    print(f"\n3D Array:\n{array_3d}")
    print(f"3D Shape: {array_3d.shape}")


def demonstrate_broadcasting() -> None:
    """Demonstrate NumPy broadcasting."""
    numbers = np.array([10, 20, 30])
    broadcast_res = numbers + 5

    print(f"\nBroadcasting: {broadcast_res}")


def demonstrate_vectorised_operations() -> None:
    """Perform arithmetic operations without Python loops."""
    array_a = np.array([1, 2, 3, 4])
    array_b = np.array([5, 6, 7, 8])

    print(f"\nAddition: {array_a + array_b}")
    print(f"Subtraction: {array_a - array_b}")
    print(f"Multiplication: {array_a * array_b}")
    print(f"Division: {array_b / array_a}")
    print(f"Square: {array_a**2}")


def demonstrate_matrix_multiplication() -> None:
    """Perform matrix multiplication using NumPy."""
    matrix_a = np.array([[1, 2], [3, 4]])
    matrix_b = np.array([[5, 6], [7, 8]])

    matmul_res = np.matmul(matrix_a, matrix_b)

    print(f"\nMatrix Multiplication:\n{matmul_res}")


def demonstrate_statistics() -> None:
    """Calculate mean, standard deviation, and correlation from Iris CSV."""
    project_folder = Path(__file__).resolve().parent.parent
    csv_path = project_folder / "data" / "iris.csv"

    try:
        data = np.loadtxt(
            csv_path,
            delimiter=",",
            skiprows=1,
            usecols=(0, 1, 2, 3),
        )
    except FileNotFoundError:
        print(f"Error: Dataset not found at {csv_path}")
        return

    mean_values = np.mean(data, axis=0)
    std_values = np.std(data, axis=0, ddof=1)
    correlation_matrix = np.corrcoef(data, rowvar=False)

    print(f"\nMean:\n{mean_values}")
    print(f"\nStandard Deviation:\n{std_values}")
    print(f"\nCorrelation Matrix:\n{correlation_matrix}")


def main() -> None:
    """Run all NumPy demonstrations."""
    demonstrate_arrays()
    demonstrate_broadcasting()
    demonstrate_vectorised_operations()
    demonstrate_matrix_multiplication()
    demonstrate_statistics()


if __name__ == "__main__":
    main()
