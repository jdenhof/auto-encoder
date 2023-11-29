import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Check if GPU is available, and use it if it is, otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Define the hyperparameters
input_dim = 784  # Example: MNIST data
hidden_dim = 256
latent_dim = 20

# Create the VAE model and move it to the selected device
vae = VAE(input_dim, hidden_dim, latent_dim).to(device)

# Define the loss function (BCE loss plus KL divergence)
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def preview(image):
        plt.imshow(image.reshape(28, 28), cmap='gray') 
        plt.axis('off') 
        plt.show()

# Create a DataLoader for your dataset (modify this for your actual data)
# Here, we create a DataLoader for random data as an example
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
test_loader_iter = iter(test_loader)
# Training loop
num_epochs = 10
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    vae.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, input_dim).to(device)  # Flatten MNIST images and move to the device
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()

    # Optionally, you can sample from the VAE and generate images
    with torch.no_grad():
        vae.eval()
        test_data, _ = next(test_loader_iter)  # Get the next image from the test set
        test_data = test_data.view(-1, input_dim).to(device)
        recon_test, _, _ = vae(test_data)
        recon_test = recon_test.view(28, 28)  # Reshape the image if needed
        plt.imshow(recon_test.cpu().numpy(), cmap='gray')
        plt.title(f'Epoch {epoch+1} Generated Image (from Test Set)')
        plt.show()