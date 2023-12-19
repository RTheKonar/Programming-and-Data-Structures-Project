import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from primal_dual_algo import chambolle_pock_algorithm
from simple_thresholding_algo import min_adverserial_risk, max_product_of_elements, almost_diagonal
from read_excel_file import read_excel_dataset
if __name__ == "__main__":
    file_path = 'your_file.xlsx'  # Replace 'your_file.xlsx' with your file path

    # Call the function to read the Excel dataset
    dataset = read_excel_dataset(file_path)

    if dataset is not None:
        data_y = dataset['y']
        data_x = dataset.drop('y', axis = 1)
        cov_matrix = np.cov(data_x, rowvar=False)
        K = np.linalg.cholesky(cov_matrix)
        #print("Covariance Matrix:")
        #print(cov_matrix)
        model = LinearRegression()
        model.fit(data_x, data_y)
        w0_hat = model.coef_
        r = 0.4 #replace r according to your 'attack budget'
        #for l2 attacks
        U, S, Vt = np.linalg.svd(cov_matrix)
        # The operator norm is the maximum singular value of Sigma
        sigma_op_norm = max(S)
        sigma_1 = sigma_op_norm**(0.5)
        eta_1 = 0.3
        eta_2 = 1/(sigma_1)
        if eta_1*eta_2*sigma_1<1:
            iteration = 100 #replace with desired no. of iterations
            result_w = chambolle_pock_algorithm(w0_hat, K, eta_1, eta_2, r, iteration)
            print("estimated w_0 from primal dual algorithm for l2 norm attacks: ")
            print(result_w)
        #for l-infty attacks
        if almost_diagonal(cov_matrix):   
            lambda_d = np.diag(cov_matrix).tolist()
            c = max_product_of_elements(lambda_d, w0_hat)
            grid_size = 10 #replace with desired grid_size
            result_w1 = min_adverserial_risk(w0_hat, c, lambda_d, r, grid_size)
            print("estimatted w_0 for l-infty attacks: ", result_w1)
        
