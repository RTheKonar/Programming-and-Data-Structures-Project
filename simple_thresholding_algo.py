import numpy as np
def almost_diagonal(matrix):
    max_element = None
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if i != j:  # Check for off-diagonal elements
                abs_val = abs(matrix[i][j])
                if max_element is None or abs_val > max_element:
                    max_element = abs_val

    # Check if the maximum off-diagonal absolute value is less than the cutoff
    if max_element is not None and max_element < 0.1:
        return True
    else:
        return False

    
def max_product_of_elements(list1, list2):
    return max(list1[i] * abs(list2[i]) for i in range(len(list1)))
def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
def l_infty_adverserial_risk(w, w0, r):
    w_inf_norm = np.linalg.norm(w-w0, ord=np.inf)
    w_1_norm = np.linalg.norm(w, ord=1)
    return (w_inf_norm+(r*w_1_norm))**2
def min_adverserial_risk(w0, c_hat, ls, r, grid_size):
    min_adverserial_risk = float('inf')
    best_w = None
    grid_values = np.linspace(0, c_hat, grid_size)
    for t in grid_values:
        # Compute w(t) for each component
        w_t = np.array([soft_threshold(w0[j], (r*t / ls[j])) for j in range(len(w0))])
        adverserial_risk = l_infty_adverserial_risk(w_t, w0, r)
        # Check if current value of t minimizes the adversarial risk
        if adverserial_risk < min_adverserial_risk:
            min_adverserial_risk = adverserial_risk
            best_w = w_t
    return best_w 
