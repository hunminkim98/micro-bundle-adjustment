import torch
from torch.autograd import grad
from tqdm import tqdm

def lm_optimize(f, X_0, theta_0, *, observations, num_cameras, num_steps=100, L_0=1e-3, device='cpu', dtype=torch.float32):
    X = X_0.clone().to(device).type(dtype).requires_grad_(True)
    theta = theta_0.clone().to(device).type(dtype).requires_grad_(True)
    optimizer = torch.optim.Adam([X, theta], lr=1e-2)

    previous_loss = None
    tolerance = 1e-6

    for step in tqdm(range(num_steps), desc="Optimization Steps"):
        optimizer.zero_grad()
        residuals = f(X, theta)
        loss = (residuals ** 2).sum()
        loss.backward()
        optimizer.step()

        tqdm.write(f"Step {step+1}/{num_steps}, Loss: {loss.item()}")

        # Early stopping based on convergence criteria
        if previous_loss is not None and abs(previous_loss - loss.item()) < tolerance:
            print(f"Converged at step {step+1}")
            break
        previous_loss = loss.item()

    return X.detach(), theta.detach()
