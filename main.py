import math, copy
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray


def read_data(file_name: str, sep=None) -> tuple:
    """
    Read and return the data from a file as well as its number of rows and columns.\n
    File data must be of format\n
    line1: no. of rows\n
    line2: no. of cols\n
    line3: empty line\n
    subsequent lines: data\n
    file_name\n\t The name of the file to read sample data from.
    sep (optional)\n\t The seperator used to split the string.
    """

    # opening .txt file
    file_handler = open(file_name, 'r')

    # reading number of rows
    rows = int(file_handler.readline())
    # reading number of columns
    cols = int(file_handler.readline())

    # skip empty line
    file_handler.readline()

    # read all lines containing the data
    data_lines = file_handler.readlines()

    # retrieve all the data from the lines and store in an array
    data_array = []
    for line in data_lines:
        data_array = data_array + line.split(sep=sep)

    # define an empty numpy array for sample data
    data = np.empty(shape=(rows, cols))

    # read data from .txt file and store in array
    index = 0
    for i in range(rows):
        for j in range(cols):
            data[i][j] = float(data_array[index])
            index += 1

    return rows, cols, data


def compute_mean_of_all_columns(rows: int, cols: int, data: ndarray) -> list:
    """
    Computes and returns a list containing means of all columns.\n
    rows\n\t The number of rows in sample data.\n
    cols\n\t The number of cols in sample data.\n
    data\n\t The sample data whose mean of columns is to be computed.
    """

    col_sums = np.zeros(shape=cols)
    for i in range(rows):
        for j in range(cols):
            col_sums[j] += data[i][j]
    col_means = col_sums / rows

    return col_means


def compute_max_of_all_columns(rows: int, cols: int, data) -> ndarray:
    """
    Computes and returns a list containing max of all columns.\n
    rows\n\t The number of rows in sample data.\n
    cols\n\t The number of cols in sample data.\n
    data\n\t The sample data whose max of columns is to be computed.
    """

    col_max = np.zeros(shape=cols)
    for i in range(rows):
        for j in range(cols):
            if data[i][j] > col_max[j]:
                col_max[j] = data[i][j]

    return col_max


def calculate_correlation(row_1: list, row_2: list) -> float:
    """
    Returns pearson correlation of 2 rows.
    """
    cols = len(row_1)

    # calculate mean of row_1 and row_2
    sum_r1, sum_r2 = 0, 0
    for i in range(cols):
        sum_r1 += float(row_1[i])
        sum_r2 += float(row_2[i])
    mean_r1, mean_r2 = (sum_r1 / 4), (sum_r2 / 4)

    numerator, denominator = 0, 0
    for i in range(cols):
        numerator += (float(row_1[i]) - mean_r1) * (float(row_2[i]) - mean_r2)

    denom_a, denom_b = 0, 0
    for i in range(cols):
        denom_a += (float(row_1[i]) - mean_r1) ** 2
        denom_b += (float(row_2[i]) - mean_r2) ** 2
    denominator = math.sqrt(denom_a * denom_b)

    return numerator / denominator


def calculate_correlation_matrix(rows: int, cols: int, data: ndarray) -> ndarray:
    """
    Computes and returns a correlation matrix of the data.\n
    rows\n\t The number of rows in sample data.\n
    cols\n\t The number of cols in sample data.\n
    data\n\t The sample data of which the correlation matrix is to be computed.
    """

    # calculate mean of all columns
    col_means = compute_mean_of_all_columns(rows=rows, cols=cols, data=data)

    # define an empty numpy array for correlated output
    correlated = np.empty(shape=(rows, rows))

    for i in range(rows):
        for j in range(rows):
            correlated[i][j] = calculate_correlation(row_1=data[i], row_2=data[j])

    return correlated


def discretize(rows_cols: int, corr_matrix: ndarray) -> ndarray:
    """
    Discretizes a correlation matrix.\n
    rows_cols\n\t The number of rows and columns in data (single variable due to square correlation matrix).\n
    data\n\t The correlation matrix which is to be discretized.
    """

    # calculate mean of all columns
    col_means = compute_mean_of_all_columns(rows=rows_cols, cols=rows_cols, data=corr_matrix)

    # define an empty numpy array for discretized output
    discretized = np.empty(shape=(rows_cols, rows_cols))

    # compute descritized output for each column based in its mean
    for j in range(rows_cols):
        for i in range(rows_cols):
            if corr_matrix[i][j] > col_means[j]:
                discretized[i][j] = 0
            else:
                discretized[i][j] = 1

    return discretized


def compute_color_coded(corr_matrix: ndarray) -> ndarray:
    """
    Computes the color coded values\n
    corr_matrix\n\t The correlation matrix whose values are to be color coded.
    """

    # calculate max of all columns
    rows_cols = corr_matrix.shape[0]
    col_max = compute_max_of_all_columns(rows=rows_cols, cols=rows_cols, data=corr_matrix)

    # define array to store color coded output
    color_coded = np.empty(shape=(rows_cols, rows_cols, 3))

    # for shade of green, setting R and B in RGB to 0
    for j in range(rows_cols):
        for i in range(rows_cols):
            color_coded[i][j] = [0, (corr_matrix[i][j] / col_max[j]) * 255, 0]

    # convert all data to int and return
    return color_coded.astype(int)


def visualize(dicretized_data: ndarray, correlated_data=None) -> None:
    """
    Plots the bitmap of the dicretized and optionally the color coded bitmap of correlated data using matplotlib.\n
    dicretized_data\n\t The data of which the bitmap is to be plotted.\n
    correlated_data (optional)\n\t The data of which the color coded bitmap is to be plotted.
    """

    if correlated_data is not None:
        color_coded = compute_color_coded(corr_matrix=correlated_data)

        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].set_title('Discretized Matrix')
        axes[0].imshow(dicretized_data, cmap='gray')
        axes[1].set_title('Color Coded Correlation Matrix')
        axes[1].imshow(color_coded)
        plt.show()

    else:
        plt.suptitle('Discretized Matrix')
        plt.imshow(dicretized_data)
        plt.set_cmap('gray')
        plt.show()


"""
FILE-1: IRIS DATA
"""
number_of_rows, number_of_columns, sample_data = read_data(file_name='Sample data-1-IRIS.TXT')
correlation_matrix = calculate_correlation_matrix(rows=number_of_rows, cols=number_of_columns, data=sample_data)
discretized_data = discretize(rows_cols=number_of_rows, corr_matrix=correlation_matrix)
visualize(dicretized_data=discretized_data, correlated_data=correlation_matrix)


"""
FILE-2: DISCRETIZED DATA
"""
# number_of_rows, number_of_columns, sample_data = read_data(file_name='Sample data-2-INPUT1.TXT')
# visualize(dicretized_data=sample_data)


"""
FILE-3: VINE DATA
"""
# number_of_rows, number_of_columns, sample_data = read_data(file_name='Sample data-3-VINE.TXT', sep=',')
# correlation_matrix = calculate_correlation_matrix(rows=number_of_rows, cols=number_of_columns, data=sample_data)
# discretized_data = discretize(rows_cols=number_of_rows, corr_matrix=correlation_matrix)
# visualize(dicretized_data=discretized_data, correlated_data=correlation_matrix)
