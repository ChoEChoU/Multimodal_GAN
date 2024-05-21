import torch
import torch.optim as optim
from torch.autograd import grad
from models import SharedGenerator, HighResGenerator, LowResGenerator, TabularGenerator, HighResCritic, LowResCritic, TabularCritic, compute_gradient_penalty
from dataloader import get_data_loader

# Training parameters
latent_dim = 100
num_classes = 10
img_shape = (1, 64, 64, 64)
batch_size = 64
epochs = 10000
lr = 0.0002
b1 = 0.5
b2 = 0.999
lambda_gp = 10

# Prepare the dataset
dataloader = get_data_loader(batch_size)

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

# Training loop
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

if __name__ == "__main__":
    main()