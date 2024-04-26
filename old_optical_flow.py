import numpy as np
import scipy.ndimage

__all__ = ["__version__", "poly_exp", "flow_iterative"]


__version__ = "1.0.0"


def poly_exp(signal, confidence, sigma):
    """
    Calculates the local polynomial expansion of a 2D signal as described by Farneback.

    Parameters
    ----------
    signal : ndarray
        2D array representing the input signal.
    confidence : ndarray
        2D array where each element indicates the confidence level of the corresponding pixel in the signal array.
    sigma : float
        Standard deviation of the Gaussian kernel used for smoothing.

    Returns
    -------
    quadratic_terms : ndarray
        3D array where each element is a 2x2 matrix representing the quadratic coefficient matrix of the polynomial expansion at each pixel.
    linear_terms : ndarray
        3D array where each element is a vector representing the linear coefficients of the polynomial expansion at each pixel.
    constant_terms : ndarray
        2D array where each element is a scalar representing the constant term of the polynomial expansion at each pixel.
    """
    # Calculate Gaussian kernel (1D since it is separable)
    kernel_size = int(4 * sigma + 1)
    positions = np.arange(-kernel_size, kernel_size + 1, dtype=int)
    gaussian_kernel = np.exp(-(positions**2) / (2 * sigma**2))

    # Construct basis matrices for X and Y dimensions
    basis_x = np.stack(
        [np.ones(gaussian_kernel.shape), positions, 
         np.ones(gaussian_kernel.shape), positions**2, 
         np.ones(gaussian_kernel.shape), positions
        ], 
         axis=-1
    )
    basis_y = np.stack(
       [np.ones(gaussian_kernel.shape), np.ones(gaussian_kernel.shape),
        positions, np.ones(gaussian_kernel.shape), 
        positions**2, positions
       ],
        axis=-1,
    )

    signal = signal[:,:,0]
    # Multiply confidence and signal for weighting
    weighted_signal = confidence * signal

    # Initialize matrices for coefficients and the solution vector
    cross_correlation_matrix = np.empty(list(signal.shape) + [basis_x.shape[-1]] * 2)
    solution_vector = np.empty(list(signal.shape) + [basis_x.shape[-1]])

    # Calculate cross-correlation matrix and solution vector using separable correlation
    weighted_basis_x = np.einsum("i,ij->ij", gaussian_kernel, basis_x)
    weighted_basis_y = np.einsum("i,ij->ij", gaussian_kernel, basis_y)

    for i in range(basis_x.shape[-1]):
        for j in range(basis_x.shape[-1]):
            cross_correlation_matrix[..., i, j] = scipy.ndimage.correlate1d(
                confidence, np.einsum("ij,ik->ijk", weighted_basis_x, basis_x)[..., i, j], axis=0, mode="constant", cval=0
            )
            cross_correlation_matrix[..., i, j] = scipy.ndimage.correlate1d(
                cross_correlation_matrix[..., i, j], np.einsum("ij,ik->ijk", weighted_basis_y, basis_y)[..., i, j], axis=1, mode="constant", cval=0
            )

        solution_vector[..., i] = scipy.ndimage.correlate1d(
            weighted_signal, weighted_basis_x[..., i], axis=0, mode="constant", cval=0
        )
        solution_vector[..., i] = scipy.ndimage.correlate1d(
            solution_vector[..., i], weighted_basis_y[..., i], axis=1, mode="constant", cval=0
        )

    # Solve for the polynomial coefficients
    polynomial_coefficients = np.linalg.solve(cross_correlation_matrix, solution_vector)

    # Extract the coefficients into separate terms
    quadratic_terms = np.empty(list(signal.shape) + [2, 2])
    quadratic_terms[..., 0, 0] = polynomial_coefficients[..., 3]
    quadratic_terms[..., 0, 1] = polynomial_coefficients[..., 5] / 2
    quadratic_terms[..., 1, 0] = quadratic_terms[..., 0, 1]
    quadratic_terms[..., 1, 1] = polynomial_coefficients[..., 4]

    linear_terms = np.empty(list(signal.shape) + [2])
    linear_terms[..., 0] = polynomial_coefficients[..., 1]
    linear_terms[..., 1] = polynomial_coefficients[..., 2]

    constant_terms = polynomial_coefficients[..., 0]

    return quadratic_terms, linear_terms, constant_terms



def flow_iterative(
    f1, f2, sigma, c1, c2, 
    sigma_flow, num_iter=1, d=None, 
    model="constant", mu=None
):
    """
    Calculates optical flow using an algorithm described by Gunnar Farneback.

    Parameters
    ----------
    f1
        First image
    f2
        Second image
    sigma
        Sigma for Gaussian kernel in polynomial expansion
    c1
        Certainty of first image
    c2
        Certainty of second image
    sigma_flow
        Sigma for Gaussian kernel in applicability window for polynomial matching
    num_iter
        Number of iterations to run (defaults to 1)
    d: (optional)
        Initial displacement field
    model: ['constant', 'affine', 'eight_param']
        Type of optical flow parametrization to use
    mu: (optional)
        Weighting term for usage of global parametrization. Defaults to
        the value recommended in Farneback's thesis

    Returns
    -------
    displacement_field
        Optical flow field. displacement_field[i, j] represents the (y, x) displacement for pixel (i, j)
    """

    # Calculate the polynomial expansion at each point in the images
    A1, B1, C1 = poly_exp(f1, c1, sigma)
    A2, B2, C2 = poly_exp(f2, c2, sigma)

    # Pixel coordinates of each point in the images
    x = np.stack(
        np.broadcast_arrays(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1])),
        axis=-1,
    ).astype(int)

    # Initialize displacement field
    if d is None:
        d = np.zeros(list(f1.shape[:2]) + [2])
        # d = np.zeros(list(f1.shape) + [2])

    # Set up applicability convolution window
    n_flow = int(4 * sigma_flow + 1)
    xw = np.arange(-n_flow, n_flow + 1)
    w = np.exp(-(xw**2) / (2 * sigma_flow**2))

    # Evaluate warp parametrization model at pixel coordinates
    if model == "constant":
        S = np.eye(2)

    elif model in ("affine", "eight_param"):
        S = np.empty(list(x.shape) + [6 if model == "affine" else 8])

        S[..., 0, 0] = 1
        S[..., 0, 1] = x[..., 0]
        S[..., 0, 2] = x[..., 1]
        S[..., 0, 3] = 0
        S[..., 0, 4] = 0
        S[..., 0, 5] = 0

        S[..., 1, 0] = 0
        S[..., 1, 1] = 0
        S[..., 1, 2] = 0
        S[..., 1, 3] = 1
        S[..., 1, 4] = x[..., 0]
        S[..., 1, 5] = x[..., 1]

        if model == "eight_param":
            S[..., 0, 6] = x[..., 0] ** 2
            S[..., 0, 7] = x[..., 0] * x[..., 1]

            S[..., 1, 6] = x[..., 0] * x[..., 1]
            S[..., 1, 7] = x[..., 1] ** 2

    else:
        raise ValueError("Invalid parametrization model")

    S_T = S.swapaxes(-1, -2)

    # Iterate convolutions to estimate the optical flow
    for _ in range(num_iter):
        # Set d~ as displacement field fit to nearest pixel (and constrain to not
        # being off image). Note we are setting certainty to 0 for points that
        # would have been off-image had we not constrained them
        d_ = d.astype(int)

        x_ = x + d_

        # x_ = np.maximum(np.minimum(x_, np.array(f1.shape) - 1), 0)

        # Constrain d~ to be on-image, and find points that would have
        # been off-image
        x_2 = np.maximum(np.minimum(x_, np.array(f1.shape[:2]) - 1), 0)
        # x_2 = np.maximum(np.minimum(x_, np.array(f1.shape) - 1), 0)
        off_f = np.any(x_ != x_2, axis=-1)
        x_ = x_2

        # Set certainty to 0 for off-image points
        c_ = c1[x_[..., 0], x_[..., 1]]
        c_[off_f] = 0

        # Calculate A and delB for each point, according to paper
        A = (A1 + A2[x_[..., 0], x_[..., 1]]) / 2
        A *= c_[
            ..., None, None
        ]  # recommendation in paper: add in certainty by applying to A and delB

        delB = -1 / 2 * (B2[x_[..., 0], x_[..., 1]] - B1) + (A @ d_[..., None])[..., 0]
        delB *= c_[
            ..., None
        ]  # recommendation in paper: add in certainty by applying to A and delB

        # Pre-calculate quantities recommended by paper
        A_T = A.swapaxes(-1, -2)
        ATA = S_T @ A_T @ A @ S
        ATb = (S_T @ A_T @ delB[..., None])[..., 0]
        # btb = delB.swapaxes(-1, -2) @ delB

        # If mu is 0, it means the global/average parametrized warp should not be
        # calculated, and the parametrization should apply to the local calculations
        if mu == 0:
            # Apply separable cross-correlation to calculate linear equation
            # for each pixel: G*d = h
            G = scipy.ndimage.correlate1d(ATA, w, axis=0, mode="constant", cval=0)
            G = scipy.ndimage.correlate1d(G, w, axis=1, mode="constant", cval=0)

            h = scipy.ndimage.correlate1d(ATb, w, axis=0, mode="constant", cval=0)
            h = scipy.ndimage.correlate1d(h, w, axis=1, mode="constant", cval=0)

            d = (S @ np.linalg.solve(G, h)[..., None])[..., 0]

        # if mu is not 0, it should be used to regularize the least squares problem
        # and "force" the background warp onto uncertain pixels
        else:
            # Calculate global parametrized warp
            G_avg = np.mean(ATA, axis=(0, 1))
            h_avg = np.mean(ATb, axis=(0, 1))
            p_avg = np.linalg.solve(G_avg, h_avg)
            d_avg = (S @ p_avg[..., None])[..., 0]

            # Default value for mu is to set mu to 1/2 the trace of G_avg
            if mu is None:
                mu = 1 / 2 * np.trace(G_avg)

            # Apply separable cross-correlation to calculate linear equation
            G = scipy.ndimage.correlate1d(A_T @ A, w, axis=0, mode="constant", cval=0)
            G = scipy.ndimage.correlate1d(G, w, axis=1, mode="constant", cval=0)

            h = scipy.ndimage.correlate1d(
                (A_T @ delB[..., None])[..., 0], w, axis=0, mode="constant", cval=0
            )
            h = scipy.ndimage.correlate1d(h, w, axis=1, mode="constant", cval=0)

            # Refine estimate of displacement field
            d = np.linalg.solve(G + mu * np.eye(2), h + mu * d_avg)

    return d
