import torch
from kornia.geometry import axis_angle_to_rotation_matrix
from optimizer import lm_optimize

def projection(X, r, t):
    R = axis_angle_to_rotation_matrix(r[None])[0]
    if len(X.shape) > 1:
        x = (R @ X.T).T + t[None]
    else:
        x = (R @ X) + t
    return x[..., :2] / x[..., [2]]

def optimize_calibrated_multi_camera(X_0, observations, theta_0, *, num_cameras, dtype=torch.float32, L_0=1e-2, num_steps=5):
    # Define the residual function
    def calibrated_residuals_multi_camera(X, theta):
        residuals = []
        for obs in observations:
            x_im = obs['x_im'].to(X.device).type(X.dtype)
            inds = obs['inds']
            cam_idx = obs['camera_idx']
            r, t = theta[cam_idx].chunk(2)
            X_points = X[inds]  # Get the 3D points for this observation
            x_im_hat = projection(X_points, r, t)
            r_im = x_im_hat - x_im
            residuals.append(r_im.reshape(-1))
        return torch.cat(residuals, dim=0)

    X_hat, theta_hat = lm_optimize(
        calibrated_residuals_multi_camera,
        X_0,
        theta_0,
        observations=observations,
        num_cameras=num_cameras,
        num_steps=num_steps,
        L_0=L_0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dtype=dtype
    )
    return X_hat, theta_hat
