
import numpy as np

def generate_gauss_mfs(X, num_mfs=3):
    """Generate Gaussian membership functions with better spread"""
    mfs = []
    for i in range(X.shape[1]):
        col = X[:, i]
        min_val, max_val = np.min(col), np.max(col)
        span = max_val - min_val
        
        # Place centers at even intervals: 1/6, 3/6, 5/6 of the span
        centers = np.linspace(min_val + span / (2 * num_mfs),
                              max_val - span / (2 * num_mfs),
                              num=num_mfs)
        
        # Set sigma to 1/6 of the full span
        sigma = span / 6
        
        col_mfs = [['gaussmf', {'mean': c, 'sigma': sigma}] for c in centers]
        mfs.append(col_mfs)
    return mfs
        
def gaussmf(x, mean, sigma):
    """Evaluate membership functions for given single input"""
    return np.exp(-((x - mean) ** 2.) / float(sigma) ** 2.)

def evaluate_mf(row_input, mf_list):
    """Evaluate membership functions for given input row"""
    return [[gaussmf(row_input[i], **mf_list[i][k][1]) for k in range(len(mf_list[i]))] for i in range(len(row_input))]

def partial_dmf(x, mf_definition, partial_parameter):
    """Partial derivative of membership function"""
    sigma = mf_definition[1]['sigma']
    mean = mf_definition[1]['mean']

    mu = np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    if partial_parameter == 'sigma':
        result = mu * ((x - mean) ** 2) / (sigma ** 3)
    elif partial_parameter == 'mean':
        result = mu * (x - mean) / (sigma ** 2)
    else:
        result = 0

    return result