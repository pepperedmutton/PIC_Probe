
@jit(nopython=True)
def smooth_density_cylindrical(rho: np.ndarray, n_passes: int) -> None:
    """Apply binomial smoothing (1-2-1) to density array.

    Args:
        rho: Charge density array (modified in-place)
        n_passes: Number of smoothing passes
    """
    n = rho.shape[0]
    if n < 3 or n_passes <= 0:
        return

    rho_new = np.empty_like(rho)
    
    for _ in range(n_passes):
        # Interior points: 0.25 * rho[j-1] + 0.5 * rho[j] + 0.25 * rho[j+1]
        for j in range(1, n - 1):
            rho_new[j] = 0.25 * rho[j - 1] + 0.5 * rho[j] + 0.25 * rho[j + 1]
            
        # Boundaries: Simple replication or 2-sided smooth?
        # Let's use simple weighting for boundaries to avoid losing mass too much
        # rho[0] = 0.5*rho[0] + 0.5*rho[1]  ?
        # Standard practice: Boundaries often not smoothed or special weight.
        # Let's just keep boundaries fixed for now as Dirichlet BCs dictate potential anyway,
        # but density at boundary affects the field nearby.
        # Let's use asymmetric 2-point smooth at edges.
        rho_new[0] = 0.666 * rho[0] + 0.334 * rho[1]
        rho_new[n - 1] = 0.666 * rho[n - 1] + 0.334 * rho[n - 2]
        
        # Copy back
        for j in range(n):
            rho[j] = rho_new[j]
