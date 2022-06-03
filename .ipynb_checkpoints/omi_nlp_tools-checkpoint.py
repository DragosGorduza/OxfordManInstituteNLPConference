import numpy as np

def mutual_information(word, n_text, n_total):
    
    # Create a matrix: for each class (row), how many articles did it appear in (col1) and not appear in (col2)
    count_matrix = np.array([[n_text["negative"][word], n_total["negative"] - n_text["negative"][word]],
                            [n_text["positive"][word], n_total["positive"] - n_text["positive"][word]]])
    
    # Store the sum of that matrix --> total number of articles in the training set
    N = count_matrix.sum()
    
    # Initialise the value we will return 
    out = 0
    
    # Iterate through columns and rows
    for col in range(2):
        for row in range(2):
            
            if count_matrix[row, col]!= 0:
                out += count_matrix[row, col] * np.log2((N*count_matrix[row, col]) / (count_matrix[row, :].sum() * count_matrix[:, col].sum()))
            
            # If the word did not appear in any tweets for either class, break the loop and return MI as 0
            else:
                return 0
    
    return (1/N) * out