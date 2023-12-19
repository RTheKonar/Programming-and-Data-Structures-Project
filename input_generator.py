import numpy as np
import pandas as pd

# Set mean and covariance matrix
mean = [0, 0]
cov_matrix = [[1, 0], [0, 1]]

# Generate data points
data = np.random.multivariate_normal(mean, cov_matrix, 500)

# Create a DataFrame
df = pd.DataFrame(data, columns=['X', 'Y'])

# Save to an Excel file
df.to_excel('C:\\Users\\aishe\\OneDrive\\Desktop\\pds project\\normal_distribution_data.xlsx', index=False)
