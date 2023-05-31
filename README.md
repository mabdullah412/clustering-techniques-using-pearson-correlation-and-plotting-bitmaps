# clustering-techniques-using-pearson-correlation-and-plotting-bitmaps

## Task 1

Calculation Correlation Matrix:
- Create a correlation matrix from the data matrix using Pearson’s correlation coefficient
- The correlation matrix will be a NxN matrix (where N is number of records in your input dataset) containing Pearson’s correlation coefficient between each of the row in data matrix

Discretize:
- Calculate median/mean of each column of the correlation matrix and set all the values in that column that are above the calculated median/mean to 1 and rest to 0

Visualize:
- Convert the discretized matrix into bitmap.
- Provide functionality for zooming.
- Display the color coded image of similarly matrix. Follow the following steps to display color coded image
  - For each column in matrix (adjacency matrix of graph), find max value.
  - Divide each value in column by max value and multiply it with 255.
  - Resulting values will be in range 0 to 255.
  - Use this value for applying green shade to pixel.

## Task 2

- Permute the Data Matrix
  - Do this by shuffling the individual rows in the dataset.
- Display color coded image of permuted Data Matrix
- Recover the image clusters using Signature technique. The method to generate the signature is as under:
  - Sum all the values in a row
  - Calculate mean of the row
  -  Multiply the Sum of the row with its Mean
  -   The above three step produces a signature for a row

- Rearrange (sort) the Similarity Matrix by signature value of each row.
- Apply Task1 on the rearranged matrix
-  Display the color coded image
