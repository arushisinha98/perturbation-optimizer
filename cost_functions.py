# import cupy as cp
import numpy as cp


def GaussianKDE(data, grid, bandwidth):
    """ Perform Gaussian KDE to transform an array to a PDF """
    # Convert data to CuPy array
    data = cp.asarray(data)
    grid = cp.asarray(grid)
    
    N = len(data)
    weights = cp.ones(N) / N
    
    # Calculate the kernels for each grid point
    scaling_factor = (1 / (cp.sqrt(2 * cp.pi) * bandwidth))
    kernels = cp.exp(-0.5 * ((grid[:, None] - data[None, :]) / bandwidth) ** 2)
    pdf = cp.sum(kernels * weights[None, :], axis=1) * scaling_factor
    pdf = cp.where(pdf <= 1e-8, 1e-8, pdf)
    return pdf
    

def KL_Divergence(p, q, dx):
    """ Calculate the KL divergence given two PDFs p and q sampled at discrete intervals dx. """
    # Ensure no zero probability values in q to avoid division by zero
    q = cp.where(q <= 1e-8, 1e-8, q)
    return cp.sum(p * cp.log(p / q) * dx)
    

def ScottsBandwidth(data):
    """
    Compute bandwidth using Scott's rule.

    Parameters:
        data (array-like): Input data.

    Returns:
        float: Bandwidth value.
    """
    std_dev = cp.std(data)
    n = len(data)
    bw = cp.power((4 * std_dev / (3 * n)), 1 / 5)
    return bw


def SilvermansBandwidth(data, num_dims=1):
    """
    Compute bandwidth using Silverman's rule.
    
    Parameters:
        data (array-like): Input data.
        num_dims (int): Number of dimensions in the data (default is 1).
    
    Returns:
        float: Bandwidth value.
    """
    std_dev = cp.std(data)
    n = len(data)
    bw = cp.power((4 * std_dev ** 5 / (3 * n)), 1 / 5) * cp.power((1 / num_dims), 1 / 5)
    return bw
    
    
def ComputeKLD(data_p, data_q, bandwidth = 1.0):
    # Convert data to CuPy arrays
    data_p = cp.asarray(data_p)
    data_q = cp.asarray(data_q)
    
    # Drop NA values
    data_p = data_p[~cp.isnan(data_p)]
    data_q = data_q[~cp.isnan(data_q)]
    
    # Replace 0 with 1e-8
    data_p = cp.where(data_p == 0, 1e-8, data_p)
    data_q = cp.where(data_q == 0, 1e-8, data_q)
    
    # Create grid points for KDE
    grid_min = min(cp.min(data_p), cp.min(data_q))
    grid_max = max(cp.max(data_p), cp.max(data_q))
    grid = cp.linspace(grid_min, grid_max, 1000)
    dx = grid[1] - grid[0]
    
    
    # Compute PDFs using the custom KDE function
    pdf_p = GaussianKDE(data_p, grid, bandwidth)
    pdf_q = GaussianKDE(data_q, grid, bandwidth)

    # Calculate KL Divergence
    kld = KL_Divergence(pdf_p, pdf_q, dx)
    return kld


def sum_of_indices(array, indices):
    """
    Compute the sum of specified indices in a CuPy array using GPU acceleration.

    Parameters:
        array (cupy.ndarray): Input array from which numbers will be summed.
        indices (list of int): List of indices whose elements will be summed.

    Returns:
        float: Sum of the elements at the specified indices.
    """
    # Convert the indices to a CuPy array
    indices_cp = cp.asarray(indices)
    
    # Validate indices
    if cp.any(indices_cp >= len(array)) or cp.any(indices_cp < 0):
        raise ValueError("Indices are out of bounds of the array.")
    
    # Sum the elements at the specified indices
    return cp.sum(array[indices_cp])


def ComputeProximity(value, target, epsilon = 1e-8):
    """
    Compute the normalized proximity measure between a value and a target value on the GPU.

    Parameters:
        value (float): Value for which the proximity is computed.
        target (float): The target value against which proximity is measured.
        epsilon (float): A small positive constant to avoid division by zero.

    Returns:
        float: Normalized proximity measure between 0 and 1.
    """
    # Convert to CuPy arrays if not already
    value = cp.asarray(value)
    target = cp.asarray(target)
    
    # Compute the normalized proximity measure
    proximity = cp.sum(cp.abs(value - target) / (cp.abs(target) + epsilon))

    return proximity
