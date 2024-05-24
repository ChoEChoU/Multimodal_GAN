import torch
import torch.nn as nn
import torch.nn.functional as F

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

class HighResCritic(nn.Module):
    def __init__(self, img_shape):
        super(HighResCritic, self).__init__()
        self.conv = nn.Sequential(
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
            nn.Conv3d(512, 1024, 3, 2, 1),
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25)
        )
        self.flatten = nn.Flatten()
        conv_output_size = 1024 * 9 * 16 * 16
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        print('input shape =', img.shape)
        conv_out = self.conv(img)
        print('conv shape =', conv_out.shape)
        flat_out = self.flatten(conv_out)
        print('flatten shape =', flat_out.shape)
        validity = self.fc(flat_out)
        print('fc shape =', validity.shape)
        return validity

class LowResCritic(nn.Module):
    def __init__(self, img_shape):
        super(LowResCritic, self).__init__()
        self.conv = nn.Sequential(
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
            nn.Conv3d(512, 1024, 3, 2, 1),
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25)
        )
        self.flatten = nn.Flatten()
        conv_output_size = 1024 * 1 * 1 * 1
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        print('input shape =', img.shape)
        conv_out = self.conv(img)
        print('conv shape =', conv_out.shape)
        flat_out = self.flatten(conv_out)
        print('flatten shape =', flat_out.shape)
        validity = self.fc(flat_out)
        print('fc shape =', validity.shape)
        return validity

class TabularCritic(nn.Module):
    def __init__(self):
        super(TabularCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(38, 128),  # Adjusted input size from 1 to 38
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        print('tabular input =', data.shape)
        out = self.model(data)  # Corrected variable name from `out` to `data`
        print('tabular linear =', out.shape)
        return out

# Gradient penalty function
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1, 1))).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0).to(real_samples.device)
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