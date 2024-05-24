import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import get_data_loader
from models import SharedGenerator, HighResGenerator, LowResGenerator, HighResCritic, LowResCritic, TabularCritic

def normalize_image(image):
    min_val = image.min()
    max_val = image.max()
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

def train_model():
    annotations_file = '/annotations.csv'
    batch_size = 4
    dataloader = get_data_loader(batch_size, annotations_file)

    latent_dim = 100
    num_classes = 2
    high_res_img_shape = (1, 257, 512, 512)  # High-resolution image shape
    low_res_img_shape = (1, 32, 32, 32)      # Low-resolution image shape

    shared_gen = SharedGenerator(latent_dim, num_classes)
    high_res_gen = HighResGenerator()
    low_res_gen = LowResGenerator()
    high_res_critic = HighResCritic(high_res_img_shape)
    low_res_critic = LowResCritic(low_res_img_shape)
    tabular_critic = TabularCritic()

    optimizer_G = optim.Adam(
        list(shared_gen.parameters()) + 
        list(high_res_gen.parameters()) + 
        list(low_res_gen.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_high_res = optim.Adam(high_res_critic.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_low_res = optim.Adam(low_res_critic.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_tabular = optim.Adam(tabular_critic.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion_adv = nn.BCELoss()
    criterion_aux = nn.CrossEntropyLoss()

    num_epochs = 100

    for epoch in range(num_epochs):
        for i, (images, clinical_data, labels) in enumerate(dataloader):
            batch_size = images.size(0)

            real_labels = torch.full((batch_size, 1), 1, dtype=torch.float)
            fake_labels = torch.full((batch_size, 1), 0, dtype=torch.float)
            
            print(f'real_labels: {real_labels}')
            print(f'fake_labels: {fake_labels}')

            noise = torch.randn(batch_size, latent_dim)
            gen_labels = torch.randint(0, num_classes, (batch_size,))

            # Update High Res Critic with high-resolution real images
            optimizer_D_high_res.zero_grad()
            high_res_images = images  # Assume these are high-resolution images
            high_res_images = normalize_image(high_res_images)
            print(f"Normalized high res real image shape: {high_res_images.shape}, min: {high_res_images.min()}, max: {high_res_images.max()}")

            # Update High Res Critic with high-resolution real images
            optimizer_D_high_res.zero_grad()
            validity_real_high_res = high_res_critic(high_res_images)
            d_loss_real_high_res = criterion_adv(validity_real_high_res, real_labels)

            fake_images_high_res = high_res_gen(shared_gen(noise, gen_labels))
            fake_images_high_res = normalize_image(fake_images_high_res)
            print(f"Fake high-res image shape: {fake_images_high_res.shape}, min: {fake_images_high_res.min()}, max: {fake_images_high_res.max()}")
            validity_fake_high_res = low_res_critic(fake_images_high_res.detach())
            d_loss_fake_high_res = criterion_adv(validity_fake_high_res, fake_labels)

            d_loss_high_res = d_loss_real_high_res + d_loss_fake_high_res
            d_loss_high_res.backward()
            optimizer_D_high_res.step()

            # Normalize low-resolution real images
            low_res_images = low_res_gen(shared_gen(noise, gen_labels))  # Generate low-resolution images
            low_res_images = normalize_image(low_res_images)
            print(f"Low res real image shape: {low_res_images.shape}, min: {low_res_images.min()}, max: {low_res_images.max()}")
            validity_real_low_res = low_res_critic(low_res_images)
            d_loss_real_low_res = criterion_adv(validity_real_low_res, real_labels)

            fake_images_low_res = low_res_gen(shared_gen(noise, gen_labels))
            fake_images_low_res = normalize_image(fake_images_low_res)
            print(f"Fake low-res image shape: {fake_images_low_res.shape}, min: {fake_images_low_res.min()}, max: {fake_images_low_res.max()}")
            validity_fake_low_res = low_res_critic(fake_images_low_res.detach())
            d_loss_fake_low_res = criterion_adv(validity_fake_low_res, fake_labels)

            d_loss_low_res = d_loss_real_low_res + d_loss_fake_low_res
            d_loss_low_res.backward()
            optimizer_D_low_res.step()

            # Update Tabular Critic
            optimizer_D_tabular.zero_grad()
            validity_real_tabular = tabular_critic(clinical_data)
            d_loss_real_tabular = criterion_adv(validity_real_tabular, real_labels)

            validity_fake_tabular = tabular_critic(clinical_data)
            d_loss_fake_tabular = criterion_adv(validity_fake_tabular, fake_labels)

            d_loss_tabular = d_loss_real_tabular + d_loss_fake_tabular
            d_loss_tabular.backward()
            optimizer_D_tabular.step()

            # Update Generators
            optimizer_G.zero_grad()
            validity_fake_high_res = low_res_critic(fake_images_high_res)
            validity_fake_low_res = low_res_critic(fake_images_low_res)
            validity_fake_tabular = tabular_critic(clinical_data)
            g_loss_high_res = criterion_adv(validity_fake_high_res, real_labels)
            g_loss_low_res = criterion_adv(validity_fake_low_res, real_labels)
            g_loss_tabular = criterion_adv(validity_fake_tabular, real_labels)
            g_loss = g_loss_high_res + g_loss_low_res + g_loss_tabular
            g_loss.backward()
            optimizer_G.step()

            if i % 50 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], '
                      f'D High Res Loss: {d_loss_high_res.item():.4f}, D Low Res Loss: {d_loss_low_res.item():.4f}, D Tabular Loss: {d_loss_tabular.item():.4f}, G Loss: {g_loss.item():.4f}')

if __name__ == "__main__":
    train_model()