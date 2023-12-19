import numpy as np
def proj_l2_ball(v):
    #Projection onto the l2 ball.
    norm_v = np.linalg.norm(v)
    if norm_v <= 1:
        return v
    else:
        return v/norm_v
def prox_dual_norm(v, r):
    #Proximal operator for the dual norm.
    norm_v = np.linalg.norm(v)
    return v*np.maximum(0, 1-(r/norm_v))
def chambolle_pock_algorithm(w0, K, eta1, eta2, r, num_iterations):
    # Initialize variables
    d = len(w0)
    z = np.zeros(d)
    w = np.zeros(d)
    u = np.zeros(d)

    for t in range(num_iterations):
        # Update z
        z = proj_l2_ball(z + eta2 * np.dot(K, u - w0))
        

        # Update w using the proximal operator
        w1 = prox_dual_norm(w - eta1 * np.dot(K, z), eta1*r)

        # Update u
        u = 2 * w1 - w
        w = w1

    return w
