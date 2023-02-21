import numpy as np


def pca(indep_var, k):
    """
    Returns a matrix with "new" independent variables constructed from the given independent variables.
    The matrix can be cut down to the more important "new" independent variables.

    Args:
        indep_var (matrix): independent variables (X) values where the columns contain the data points
                            for one independent variable
        k: hyperparameter that determines how many features to keep.

    Returns:
        new_z (matrix): set of decorrelated features based on original independent variables. Columns
                        represent independent variables.

    """
    # Subtract the mean of the each column (list within the nested list) from each entry
    for item in indep_var:                  # will switch to NumPy arrays
        mean = sum(item) / len(item)
        for num in item:
            num += mean

    # Create the Covariance matrix
    covariance_matrix = np.dot(z.transpose(), indep_var)

    # Find the eigenvectors and sort by greatest to smallest
    p = np.linalg.svd(covariance_matrix)

    # Standardize
    new_z = np.dot(z, p)

    # Keep k features/columns
    new_var = []                            # will switch to NumPy arrays
    for row in new_z:
        temp_row = []
        for i in range(k):
            row.append(row[i])
        new_var.append(temp_row)

    return new_var
