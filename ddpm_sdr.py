import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import logging

# import sys
# import os
# os.chdir("F:\\office\\research\\diff_dim_reduc\\ddpm_sdr")
# os.getcwd()

# sys.path.append(os.path.abspath(os.path.dirname("modules.py")))

# from modules import UNet_conditional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def generate_data(n=100, p=10, sigma=1.0, b=None, A=None):
#     """
#     Generate synthetic data based on the model Y = cos(b^T X) + epsilon.

#     Parameters:
#     - n: number of samples
#     - p: dimensionality of X
#     - sigma: standard deviation of the noise epsilon
#     - b: coefficients for the linear combination in the cosine function, numpy array of shape (p,)
    
#     Returns:
#     - X: feature matrix
#     - Y: target values
#     """
#     np.random.seed(42)
#     if b is None:
#         b = np.random.randn(p)  # Random coefficients if not provided
#     if A is None:
#         m = p + 5
#         A = np.zeros((m, p))
#         np.fill_diagonal(A[:p, :], 1)  # Fill the top p x p portion with an identity matrix
#         A[p:, :] = np.random.rand(m - p, p)  # Fill the remaining portion with random values
#     # Generate high-dimensional data X
    
#     Z = np.random.multivariate_normal(np.zeros(p), np.eye(p), n)
#     X = Z @ A.T
#     epsilon = np.random.normal(0, sigma, n)
#     #Y = np.cos(X @ b) + epsilon
#     Y = 2 * np.log(np.abs(X @ b)) + 0.5 * epsilon
#     #logging.info("Data generated with n samples and p features.")
#     return X, Y, A



np.random.seed(42)
torch.manual_seed(42)

# ----- Step 1: Generate Data -----

def generate_data(n=5000, d=10, D=50, sigma=1.0, b=None, A=None):
    
    Z = np.random.multivariate_normal(np.zeros(d), np.eye(d), n)  # Z: (n, d)
    print(Z.shape)
    # A = np.random.randn(D, d)  # A: (D, d)
    # b = np.random.randn(D)  # b: (D,)
    if A is None:    
        A = np.random.randn(D, d)  # A: (D, d)
    print(A.shape)
    if b is None:
        b = np.random.randn(D)  # b: (D,)
    print(b.shape)


    X = Z @ A.T  # X: (n, D)
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-6  # 避免除 0

    # 标准化
    X_norm = (X - X_mean) / X_std

    print(X.shape)
    epsilon = np.random.normal(0, sigma, n)
    Y = 2 * np.log(np.abs(X_norm @ b) + 1e-6) + 0.5 * epsilon  # avoid log(0)
    print(Y.shape)
    return torch.tensor(X_norm, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32), torch.tensor(Z, dtype=torch.float32), A, b
    #return X, Y, Z, A, b

X, Y, Z, A, b = generate_data(n=5000, d=10, D=50, sigma=1.0)


# ----- Step 2: Define the Model -----
# Dimensionality Reduction Network f: X -> Z_hat
class DimReducer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DimReducer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


# Lightweight 1D U-Net style conditional model
class SimpleUNet1D(nn.Module):
    def __init__(self, input_dim, out_dim, time_embed_dim=32):
        super(SimpleUNet1D, self).__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.label_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.input_proj = nn.Linear(input_dim  + time_embed_dim, 64)
        self.down1 = nn.Linear(64, 128)
        self.down2 = nn.Linear(128, 256)
        self.up1 = nn.Linear(256, 128)
        self.up2 = nn.Linear(128, 64)
        self.output_proj = nn.Linear(64, out_dim)
        self.input_dim = input_dim
        self.out_dim = out_dim


    def forward(self, x, t, y=None):
        t_embed = self.time_embed(t.float().view(-1, 1))
        if y is not None:
            label_input = y.view(-1, 1) if y.dim() == 1 else y
            t_embed += self.label_embed(label_input)
        
        input_cat = torch.cat([x, t_embed], dim=-1)
        x1 = F.relu(self.input_proj(input_cat))
        x2 = F.relu(self.down1(x1))
        x3 = F.relu(self.down2(x2))
        x4 = F.relu(self.up1(x3))
        x5 = F.relu(self.up2(x4 + x2))
        out = self.output_proj(x5 + x1)
        return out

# Composite Model: f(X) -> Z_hat -> DDPM model
class CompositeModel(nn.Module):
    def __init__(self, dim_redc, unet):
        super(CompositeModel, self).__init__()
        self.dim_redc = dim_redc
        self.unet = unet

    def forward(self, x, t, y=None):
        z_hat = self.dim_redc(x)
        return self.unet(z_hat, t, y), z_hat


class ConditionalDDPM:
    def __init__(self, model, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device='cuda', out_dim=50):
        logging.info("Initializing the DDPM model")
        self.model = model.to(device)
        self.noise_steps = noise_steps
        self.device = device
        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.out_dim = out_dim

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).to(self.device)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise
    
    @torch.no_grad()
    def sample(self, y, cfg_scale=0):
        logging.info("Starting the sampling process")
        self.model.eval()
        n = len(y)
        input_dim = self.model.unet.out_dim  # get dim from final layer of f
        x = torch.randn((n, input_dim), device=self.device)
        y = y.to(self.device)
        for i in reversed(range(1, self.noise_steps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            predicted_noise, _ = self.model(x, t, y)
            if cfg_scale > 0:
                uncond_predicted_noise, _ = self.model(x, t, torch.zeros_like(y))
                predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
            alpha = self.alpha[t][:, None]
            alpha_hat = self.alpha_hat[t][:, None]
            beta = self.beta[t][:, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            # x = 1 / torch.sqrt(alpha) * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * predicted_noise) + torch.sqrt(beta) * noise
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            if i % 200 == 0:
                print(f"Step {i}: x norm = {x.norm().item():.2f}, pred_noise norm = {predicted_noise.norm().item():.2f}")

        logging.info("Sampling completed")
        return x.cpu().numpy()
    
    def train(self, dataloader, epochs=100, lr=1e-4):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        mse_loss = nn.MSELoss()
        loss_history = []  # List to store loss history

        logging.info("Training started")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for x, y, z in dataloader:
                x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
                t = self.sample_timesteps(x.size(0))
                x_t, noise = self.noise_images(x, t)
                if np.random.rand() < 0.1:
                    y = None
                predicted_noise, _ = self.model(x_t, t, y)
                #print("predicted_noise shape:", predicted_noise.shape)
                #print("noise shape:", noise.shape)
                loss = mse_loss(noise, predicted_noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
            epoch_loss = total_loss / len(dataloader.dataset)
            loss_history.append(epoch_loss)  # Append the average loss over the epoch
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        return loss_history
    
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


def plot_loss_history(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
# The rest of your code remains the same.

# p = 10
# b = np.array([1, -0.5, 0.3, -1.2, 1.1, 0.7, -0.3, 0.8, -0.2, 0.5, 1, 1, 1, 1, 1])  # Custom coefficients
# X, Y, Z = generate_data(n=100, p=p, sigma=0.001, b=b)


batch_size = 64
dataset = TensorDataset(X, Y, Z)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
f_net = DimReducer(input_dim=50, output_dim=10, hidden_dim=32).to(device)
unet = SimpleUNet1D(input_dim=10, out_dim=50).to(device)
composite = CompositeModel(f_net, unet)

ddpm = ConditionalDDPM(model=composite, device=device)
loss_history = ddpm.train(dataloader, epochs=30)

# Plot loss
plot_loss_history(loss_history)

f_net.eval()
with torch.no_grad():
    Z_hat = f_net(torch.tensor(X, dtype=torch.float32).to(device)).cpu()
Z_true = torch.tensor(Z, dtype=torch.float32)

# MSE per sample
z_mse = ((Z_hat - Z_true) ** 2).mean(dim=1)
print("Ẑ vs Z MSE (mean):", z_mse.mean().item())

# 使用训练好的模型采样
sampled_X = ddpm.sample(torch.tensor(Y[:100], dtype=torch.float32))  # 生成前100个样本对应的 X
X_true = X[:100].cpu().numpy()  # 原始 X
mse_x = ((sampled_X - X_true) ** 2).mean(axis=1)
print("平均生成 X vs 原始 X 的 MSE:", mse_x.mean())


# Example usage
#X, Y = generate_data(n=100, p=10, sigma=1.0)
# dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# model = SimpleMLP(input_dim=p+6, output_dim=p)
# u_cond = UNet_conditional()  # Adjust input dimensions
# ddpm = ConditionalDDPM(model)
# loss_history = ddpm.train(dataloader, epochs=50)  # Reduced epochs for quick testing

# Sample new data
# y_grid = np.linspace(-5, 5, 100)  # Adjust range and count as necessary
# y_tensor = torch.tensor(y_grid, dtype=torch.float32).to('cuda')
# samples = ddpm.sample(y_tensor)
# print(samples)

# plot_data_comparison(X ,Y, samples, y_tensor.cpu())  # Adjusted to plot based on your data type


