import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_data(n=100, p=10, sigma=1.0, b=None):
    """
    Generate synthetic data based on the model Y = cos(b^T X) + epsilon.

    Parameters:
    - n: number of samples
    - p: dimensionality of X
    - sigma: standard deviation of the noise epsilon
    - b: coefficients for the linear combination in the cosine function, numpy array of shape (p,)
    
    Returns:
    - X: feature matrix
    - Y: target values
    """
    np.random.seed(42)
    if b is None:
        b = np.random.randn(p)  # Random coefficients if not provided
    X = np.random.multivariate_normal(np.zeros(p), np.eye(p), n)
    epsilon = np.random.normal(0, sigma, n)
    #Y = np.cos(X @ b) + epsilon
    Y = 1 * np.log(np.abs(X @ b)) + epsilon
    logging.info("Data generated with n samples and p features.")
    return X, Y

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, t, y):
        x = torch.cat([x, y.unsqueeze(-1)], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



class ConditionalDDPM:
    def __init__(self, model, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        logging.info("Initializing the DDPM model")
        self.model = model.to(device)
        self.noise_steps = noise_steps
        self.device = device
        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).to(self.device)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise
    
    @torch.no_grad()
    def sample(self, y, cfg_scale=3):
        logging.info("Starting the sampling process")
        n = len(y)
        x = torch.randn((n, self.model.fc1.in_features - 1)).to(self.device)
        for i in reversed(range(1, self.noise_steps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x, t, y)
            if cfg_scale > 0:
                uncond_predicted_noise = self.model(x, t, torch.zeros_like(y))
                predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
            alpha = self.alpha[t][:, None]
            alpha_hat = self.alpha_hat[t][:, None]
            beta = self.beta[t][:, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * predicted_noise) + torch.sqrt(beta) * noise
        logging.info("Sampling completed")
        return x.cpu().numpy()
    
    def train(self, dataloader, epochs=100, lr=1e-4):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        mse_loss = nn.MSELoss()
        logging.info("Training started")
        for epoch in range(epochs):
            self.model.train()
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                t = self.sample_timesteps(x.size(0))
                x_t, noise = self.noise_images(x, t)
                predicted_noise = self.model(x_t, t, y)
                loss = mse_loss(noise, predicted_noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

def plot_data_comparison(X, Y, X_sampled, Y_new, feature_index=0):
    plt.figure(figsize=(10, 5))
    # Sorting the data by Y values for a more coherent line plot
    sorted_indices = np.argsort(X[:, feature_index])
    sorted_Y = Y[sorted_indices]
    sorted_X = X[sorted_indices, feature_index]
    
    # Sorting the new Y and corresponding X_sampled for line plot coherence
    new_sorted_indices = np.argsort(X_sampled[:, feature_index])
    new_sorted_Y = Y_new[new_sorted_indices]
    new_sorted_X_sampled = X_sampled[new_sorted_indices, feature_index]
    
    plt.scatter(sorted_X, sorted_Y, label='Original X', linestyle='-', marker='o', alpha=0.7, color='blue')
    plt.scatter(new_sorted_X_sampled, new_sorted_Y, label='Sampled X', linestyle='-', marker='x', alpha=0.7, color='red')
    plt.ylabel('Y Values')
    plt.xlabel(f'X Feature {feature_index}')
    plt.title('Comparison of Original and Sampled X given Y')
    plt.legend()
    plt.grid(True)
    plt.show()

# The rest of your code remains the same.

p = 10
b = np.array([1, -0.5, 0.3, -1.2, 1.1, 0.7, -0.3, 0.8, -0.2, 0.5])  # Custom coefficients
X, Y = generate_data(n=1000, p=p, sigma=0.001, b=b)
# Example usage
#X, Y = generate_data(n=100, p=10, sigma=1.0)
dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

model = SimpleMLP(input_dim=11, output_dim=10)  # Adjust input dimensions
ddpm = ConditionalDDPM(model)
ddpm.train(dataloader, epochs=10)  # Reduced epochs for quick testing

# Sample new data
y_grid = np.linspace(-5, 5, 100)  # Adjust range and count as necessary
y_tensor = torch.tensor(y_grid, dtype=torch.float32).to('cuda')
samples = ddpm.sample(y_tensor)
print(samples)

plot_data_comparison(X ,Y, samples, y_tensor.cpu())  # Adjusted to plot based on your data type
