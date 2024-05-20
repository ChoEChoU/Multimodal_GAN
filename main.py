import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np

# Custom dataset class (modify as needed for actual CT and clinical data)
class CustomMedicalDataset(Dataset):
    def __init__(self, ct_data, clinical_data, labels, transform=None):
        self.ct_data = ct_data
        self.clinical_data = clinical_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.ct_data)

    def __getitem__(self, idx):
        ct_sample = self.ct_data[idx]
        clinical_sample = self.clinical_data[idx]
        label = self.labels[idx]
        if self.transform:
            ct_sample = self.transform(ct_sample)
        return ct_sample, clinical_sample, label

# Residual block for the tabular generator
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

# Shared generator
class SharedGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(SharedGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.l1 = nn.Linear(latent_dim + num_classes, 128 * 8 * 8 * 8)
        self.conv_blocks = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, 8, 8, 8)
        img = self.conv_blocks(out)
        return img

# High-resolution generator
class HighResGenerator(nn.Module):
    def __init__(self):
        super(HighResGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.LayerNorm([128, 16, 16, 16]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(128, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Low-resolution generator
class LowResGenerator(nn.Module):
    def __init__(self):
        super(LowResGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Tabular generator
class TabularGenerator(nn.Module):
    def __init__(self):
        super(TabularGenerator, self).__init__()
        self.model = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# High-resolution critic
class HighResCritic(nn.Module):
    def __init__(self, img_shape):
        super(HighResCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(img_shape[0], 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv3d(64, 128, 3, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv3d(128, 256, 3, 2, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv3d(256, 512, 3, 2, 1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4 * 4, 1)
        )

    def forward(self, img):
        return self.model(img)

# Low-resolution critic
class LowResCritic(nn.Module):
    def __init__(self, img_shape):
        super(LowResCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(img_shape[0], 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv3d(64, 128, 3, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv3d(128, 256, 3, 2, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4 * 4, 1)
        )

    def forward(self, img):
        return self.model(img)

# Tabular critic
class TabularCritic(nn.Module):
    def __init__(self):
        super(TabularCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )

    def forward(self, data):
        return self.model(data)

# Gradient penalty function
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1, 1))).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(real_samples.device)
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Training parameters
latent_dim = 100
num_classes = 10
img_shape = (1, 64, 64, 64)  # Adjust as per your 3D CT image size
batch_size = 64
epochs = 10000
lr = 0.0002
b1 = 0.5
b2 = 0.999
lambda_gp = 10

# Prepare the dataset (modify as needed for actual CT and clinical data)
ct_data = np.random.randn(1000, *img_shape).astype(np.float32)  # Placeholder
clinical_data = np.random.randn(1000, 1).astype(np.float32)  # Placeholder
labels = np.random.randint(0, num_classes, 1000)  # Placeholder

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = CustomMedicalDataset(ct_data, clinical_data, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
shared_generator = SharedGenerator(latent_dim, num_classes)
high_res_generator = HighResGenerator()
low_res_generator = LowResGenerator()
tabular_generator = TabularGenerator()
high_res_critic = HighResCritic(img_shape)
low_res_critic = LowResCritic(img_shape)
tabular_critic = TabularCritic()

# Optimizers
optimizer_G = optim.Adam(
    list(shared_generator.parameters()) + 
    list(high_res_generator.parameters()) + 
    list(low_res_generator.parameters()) + 
    list(tabular_generator.parameters()), lr=lr, betas=(b1, b2)
)
optimizer_D_high_res = optim.Adam(high_res_critic.parameters(), lr=lr, betas=(b1, b2))
optimizer_D_low_res = optim.Adam(low_res_critic.parameters(), lr=lr, betas=(b1, b2))
optimizer_D_tabular = optim.Adam(tabular_critic.parameters(), lr=lr, betas=(b1, b2))

# Training
for epoch in range(epochs):
    for i, (ct_imgs, clinical_data, labels) in enumerate(dataloader):

        batch_size = ct_imgs.size(0)

        # Configure input
        real_ct_imgs = ct_imgs.type(torch.FloatTensor)
        real_clinical_data = clinical_data.type(torch.FloatTensor)
        labels = labels.type(torch.LongTensor)

        valid = torch.ones(batch_size, 1, requires_grad=False)
        fake = torch.zeros(batch_size, 1, requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        gen_labels = torch.randint(0, num_classes, (batch_size,))
        shared_features = shared_generator(z, gen_labels)
        gen_high_res_imgs = high_res_generator(shared_features)
        gen_low_res_imgs = low_res_generator(shared_features)
        gen_tabular_data = tabular_generator(shared_features.view(batch_size, -1))

        g_loss_high_res = -torch.mean(high_res_critic(gen_high_res_imgs))
        g_loss_low_res = -torch.mean(low_res_critic(gen_low_res_imgs))
        g_loss_tabular = -torch.mean(tabular_critic(gen_tabular_data))

        g_loss = (g_loss_high_res + g_loss_low_res + g_loss_tabular) / 3

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D_high_res.zero_grad()
        d_loss_high_res_real = -torch.mean(high_res_critic(real_ct_imgs))
        d_loss_high_res_fake = torch.mean(high_res_critic(gen_high_res_imgs.detach()))
        gradient_penalty_high_res = compute_gradient_penalty(high_res_critic, real_ct_imgs.data, gen_high_res_imgs.data)
        d_loss_high_res = d_loss_high_res_real + d_loss_high_res_fake + lambda_gp * gradient_penalty_high_res
        d_loss_high_res.backward()
        optimizer_D_high_res.step()

        optimizer_D_low_res.zero_grad()
        d_loss_low_res_real = -torch.mean(low_res_critic(real_ct_imgs))
        d_loss_low_res_fake = torch.mean(low_res_critic(gen_low_res_imgs.detach()))
        gradient_penalty_low_res = compute_gradient_penalty(low_res_critic, real_ct_imgs.data, gen_low_res_imgs.data)
        d_loss_low_res = d_loss_low_res_real + d_loss_low_res_fake + lambda_gp * gradient_penalty_low_res
        d_loss_low_res.backward()
        optimizer_D_low_res.step()

        optimizer_D_tabular.zero_grad()
        d_loss_tabular_real = -torch.mean(tabular_critic(real_clinical_data))
        d_loss_tabular_fake = torch.mean(tabular_critic(gen_tabular_data.detach()))
        gradient_penalty_tabular = compute_gradient_penalty(tabular_critic, real_clinical_data.data, gen_tabular_data.data)
        d_loss_tabular = d_loss_tabular_real + d_loss_tabular_fake + lambda_gp * gradient_penalty_tabular
        d_loss_tabular.backward()
        optimizer_D_tabular.step()

    print(f"[Epoch {epoch}/{epochs}] [D high res loss: {d_loss_high_res.item()}] [D low res loss: {d_loss_low_res.item()}] [D tabular loss: {d_loss_tabular.item()}] [G loss: {g_loss.item()}]")

    # Save generated samples every 200 epochs
    if epoch % 200 == 0:
        save_image(gen_high_res_imgs.data[:25], f"images/high_res_{epoch}.png", nrow=5, normalize=True)
        save_image(gen_low_res_imgs.data[:25], f"images/low_res_{epoch}.png", nrow=5, normalize=True)