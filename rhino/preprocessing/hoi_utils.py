import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# From https://gist.github.com/CarloNicolini/7118015?permalink_comment_id=4599314#gistcomment-4599314
def estimate_similarity_transformation(
    source: np.ndarray, target: np.ndarray, with_scaling=True
) -> np.ndarray:
    """
    Estimate similarity transformation (rotation, scale, translation) from source to target (such as the Sim3 group).
    """
    k, n = source.shape
    # print(source, target)

    mx = source.mean(axis=1)
    my = target.mean(axis=1)
    source_centered = source - np.tile(mx, (n, 1)).T
    target_centered = target - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(source_centered**2, axis=0))
    sy = np.mean(np.sum(target_centered**2, axis=0))

    Sxy = (target_centered @ source_centered.T) / n
    rank = np.linalg.matrix_rank(Sxy)
    if rank < k:
        # raise ValueError("Failed to estimate similarity transformation")
        # print("Failed to estimate similarity transformation")
        return None, None, None, False

    U, D, Vt = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = Vt.T

    S = np.eye(k)
    if np.linalg.det(Sxy) < 0:
        S[k - 1, k - 1] = -1

    R = U @ S @ V.T

    if with_scaling:
        s = np.trace(np.diag(D) @ S) / sx
        t = my - s * (R @ mx)
    else:
        t = my - (R @ mx)
        s = 1.0

    return R, s, t, True


# Call RANSAC on the two sets of points
def ransac(src, dst, threshold=0.05, max_iter=1000, verbose=True, seed=42):
    assert src.shape == dst.shape
    n, dim = src.shape
    rng = np.random.RandomState(seed)
    best_inliers = []
    best_score = 0
    for _ in range(max_iter):
        idx = rng.choice(n, 4, replace=False)
        src_sample = src[idx]
        dst_sample = dst[idx]
        R, s, t, success = estimate_similarity_transformation(
            src_sample.T, dst_sample.T, with_scaling=True
        )
        if not success:
            continue
        new_src = (s * R @ src.T + t[:, None]).T
        diff = np.linalg.norm(new_src - dst, axis=1)
        inliers = np.where(diff < threshold)[0]
        score = len(inliers)
        if score > best_score:
            best_score = score
            best_inliers = inliers
            # print(score, best_inliers)
            if verbose:
                print("Max # inliers:", best_score)
                print("Indices of inliers:")
                print(best_inliers)

    # use inlier to fit the final transformation
    src_inliers = src[best_inliers]
    dst_inliers = dst[best_inliers]
    R, s, t, success = estimate_similarity_transformation(
        src_inliers.T, dst_inliers.T, with_scaling=True
    )
    assert success

    return R, s, t, best_inliers


def average_transformation_matrices(T1, T2, weights=None):
    """
    Properly average two or more 4x4 camera-to-world transformation matrices.
    
    Args:
        T1, T2: 4x4 transformation matrices
        weights: Optional weights for averaging (default: [0.5, 0.5])
    
    Returns:
        T_avg: Averaged 4x4 transformation matrix
    """
    if weights is None:
        weights = [0.5, 0.5]
    
    # Extract rotation matrices and translation vectors
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]
    
    # Convert rotation matrices to scipy Rotation objects
    rot1 = R.from_matrix(R1)
    rot2 = R.from_matrix(R2)
    
    # Use SLERP (Spherical Linear Interpolation) to interpolate between rotations
    key_times = [0, 1]
    key_rots = R.concatenate([rot1, rot2])
    slerp = Slerp(key_times, key_rots)
    
    # Interpolate at the weighted position
    R_avg = slerp(weights[1]).as_matrix()
    
    # Weighted average of translation vectors
    t_avg = weights[0] * t1 + weights[1] * t2
    
    # Construct averaged transformation matrix
    T_avg = np.eye(4)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg
    
    return T_avg

