import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from models.ctgan.data_sampler import DataSampler
from models.ctgan.data_transformer import DataTransformer
from models.ctgan.synthesizers.ctgan import *
from models.mmcgan import Img_Discriminator, Generator, gumbel_softmax, inf_train_gen
from models.hagan.Model_HA_GAN_256 import Encoder, Sub_Encoder
from models.ctgan.synthesizers.ctgan import Discriminator as Tab_Discriminator

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
# from tensorboardX import SummaryWriter

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

device = torch.device('cuda:1') if torch.cuda.is_available() else 'cpu'
num_epochs = 1
batch_size = 10

# configs for HA-GAN
workers = 8
img_size = 256
# num_iter = 80000
# log_iter = 20
# continue_iter = 0
latent_dim = 1024
g_iter = 1
lr_g = 0.0001
lr_d = 0.0004
lr_e = 0.0001
# data_dir = 'set to your data path'
exp_name = 'MMCGAN_run'
# fold = 0

# configs for CTGAN
embedding_dim = 128
generator_dim = (256, 256)
discriminator_dim = (256, 256)
generator_lr = 2e-4
generator_decay = 1e-6
discriminator_lr = 2e-4
discriminator_decay = 1e-6
discriminator_steps = 1
log_frequency = True
pac = 10

# configs for conditional generation (num_class=0 for unconditional generation)
lambda_class = 0.1
num_class = 4


# Image dataset
ct_images = np.load('./utils/data/ct_images.npy')
ct_images = torch.tensor(ct_images, dtype=torch.float32).unsqueeze(1) # (100, 1, 256, 256, 256)
labels = np.random.randint(0, num_class, ct_images.shape[0])
labels = torch.tensor(labels, dtype=torch.long) # (100,)

train_set = TensorDataset(ct_images, labels)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=workers,
                          drop_last=True, shuffle=False)

gen_load = inf_train_gen(train_loader)

# Tabular dataset
train_data = pd.read_csv('./utils/data/clinical_data.csv')
discrete_columns = ['categorical_feature_1',
                    'categorical_feature_2',
                    'categorical_feature_3',
                    'categorical_feature_4',
                    'categorical_feature_5',
                    'categorical_feature_6',
                    'categorical_feature_7',
                    'binary_feature_1',
                    'binary_feature_2',
                    'binary_feature_3',
                    'binary_feature_4',
                    'binary_feature_5',
                    'binary_feature_6',
                    'binary_feature_7',
                    'binary_feature_8',
                    'binary_feature_9']


transformer = DataTransformer()
transformer.fit(train_data, discrete_columns)
train_data = transformer.transform(train_data)

data_sampler = DataSampler(
    train_data,
    transformer.output_info_list,
    log_frequency
)

data_dim = transformer.output_dimensions

G = Generator(
    embedding_dim,
    generator_dim,
    data_dim,
    data_sampler.dim_cond_vec(),
    mode='train',
    latent_dim=latent_dim
).to(device)

Img_D = Img_Discriminator(num_class=num_class, device=device).to(device)
Tab_D = Tab_Discriminator(
    data_dim + data_sampler.dim_cond_vec(),
    discriminator_dim,
    pac=pac
).to(device)

E = Encoder().to(device)
Sub_E = Sub_Encoder(latent_dim=latent_dim).to(device)

params = []
for n, p in G.named_parameters():
    if 'tab' in n:
        pass
    else:
        params.append(p)

img_G_params = params
tab_G_params = G.tab_G.parameters()

g_optim_img = optim.Adam(img_G_params, lr=lr_g, betas=(0.0,0.999), eps=1e-8)
d_optim_img = optim.Adam(Img_D.parameters(), lr=lr_d, betas=(0.0,0.999), eps=1e-8)
e_optim_img = optim.Adam(E.parameters(), lr=lr_e, betas=(0.0,0.999), eps=1e-8)
sub_e_optim_img = optim.Adam(Sub_E.parameters(), lr=lr_e, betas=(0.0,0.999), eps=1e-8)

g_optim_tab = optim.Adam(
            tab_G_params,
            lr=generator_lr,
            betas=(0.5, 0.9),
            weight_decay=generator_decay
        )
d_optim_tab = optim.Adam(
            Tab_D.parameters(),
            lr=discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=discriminator_decay
        )


# G = nn.DataParallel(G)
# Img_D = nn.DataParallel(Img_D)
# Tab_D = nn.DataParallel(Tab_D)
# E = nn.DataParallel(E)
# Sub_E = nn.DataParallel(Sub_E)

G.train()
Img_D.train()
Tab_D.train()
E.train()
Sub_E.train()

real_y = torch.ones((batch_size, 1)).to(device)
fake_y = torch.zeros((batch_size, 1)).to(device)

loss_f = nn.BCEWithLogitsLoss()
loss_mse = nn.L1Loss()

fake_labels = torch.zeros((batch_size, 1)).to(device)
real_labels = torch.ones((batch_size, 1)).to(device)

# summary_writer = SummaryWriter('./checkpoint/' + exp_name)


for p in Img_D.parameters():  
    p.requires_grad = False
for p in Tab_D.parameters():
    p.requires_grad = False
for p in G.parameters():  
    p.requires_grad = False
for p in E.parameters():  
    p.requires_grad = False
for p in Sub_E.parameters():  
    p.requires_grad = False


steps_per_epoch = max(len(train_data) // batch_size, 1)

epoch_iterator = tqdm(range(num_epochs), desc=f'Epoch 0/{num_epochs}')
for epoch in epoch_iterator:
    
    step_iterator = tqdm(range(steps_per_epoch), desc=f'Step 0/{steps_per_epoch}')
    for id_ in step_iterator:

        for n in range(discriminator_steps):

            ##################################################
            # Train Discriminator (D^H and D^L)
            ##################################################
            
            for p in Img_D.parameters():  
                p.requires_grad = True
            for p in Tab_D.parameters():
                p.requires_grad = True
            for p in Sub_E.parameters():  
                p.requires_grad = False

            Img_D.zero_grad()

            # random input noise for the shared generator
            noise = torch.randn((batch_size, latent_dim)).to(device)
            
            # conditional vector for the shared generator and the CTGAN core
            condvec = data_sampler.sample_condvec(batch_size)

            real_images, class_label = gen_load.__next__()
            real_images = real_images.float().to(device)
            class_label = class_label.long().to(device)
            
            # low-res full volume of real image
            real_images_small = F.interpolate(real_images, scale_factor = 0.25)
            
            # randomly select a high-res sub-volume from real image
            crop_idx = np.random.randint(0,img_size*7/8+1) # 256 * 7/8 + 1
            real_images_crop = real_images[:,:,crop_idx:crop_idx+img_size//8,:,:]

            if condvec is None: # unconditional
                c1, m1, col, opt = None, None, None, None
                real = data_sampler.sample_data(train_data, batch_size, col, opt)

                y_real_pred = Img_D(real_images_crop, real_images_small, crop_idx)
                d_real_loss = loss_f(y_real_pred, real_labels)

                # fake_images: high-res sub-volume of generated image
                # fake_images_small: low-res full volume of generated image
                fake_images, fake_images_small, fake_tab = G(noise, crop_idx=crop_idx, cond=None)
                y_fake_pred = Img_D(fake_images, fake_images_small, crop_idx)

            else: # conditional
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(device)
                m1 = torch.from_numpy(m1).to(device)

                perm = np.arange(batch_size)
                np.random.shuffle(perm)
                real = data_sampler.sample_data(train_data, batch_size, col[perm], opt[perm])
                c2 = c1[perm]

                y_real_pred, y_real_class = Img_D(real_images_crop, real_images_small, crop_idx)
                
                # GAN loss + auxiliary classifier loss
                d_real_loss = loss_f(y_real_pred, real_labels) + F.cross_entropy(y_real_class, class_label)

                fake_images, fake_images_small, fake_tab = G(noise, crop_idx=crop_idx, cond=c1)
                y_fake_pred, y_fake_class= Img_D(fake_images, fake_images_small, crop_idx)
            
            '''
                Apply proper activation function to the output of the tabular generator
            '''
            data_t = []
            st = 0

            for column_info in transformer.output_info_list:
                for span_info in column_info:
                    if span_info.activation_fn == 'tanh':
                        ed = st + span_info.dim
                        data_t.append(torch.tanh(fake_tab[:, st:ed]))
                        st = ed
                    elif span_info.activation_fn == 'softmax':
                        ed = st + span_info.dim
                        transformed = gumbel_softmax(fake_tab[:, st:ed], tau=0.2)
                        data_t.append(transformed)
                        st = ed
                    else:
                        raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

            fakeact = torch.cat(data_t, dim=1)
            real = torch.from_numpy(real.astype('float32')).to(device)

            if c1 is not None:
                fake_cat = torch.cat([fakeact, c1], dim=1)
                real_cat = torch.cat([real, c2], dim=1)
            else:
                real_cat = real
                fake_cat = fakeact
            
            y_fake = Tab_D(fake_cat)
            y_real = Tab_D(real_cat)

            pen = Tab_D.calc_gradient_penalty(
                real_cat, fake_cat, device, pac)
            loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

            d_optim_tab.zero_grad(set_to_none=False)
            pen.backward(retain_graph=True)
            loss_d.backward()
            d_optim_tab.step()
            
            d_fake_loss = loss_f(y_fake_pred, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()

            d_optim_img.step()
        

        ##################################################
        # Train Generator (G^A, G^H and G^L)
        ##################################################        

        for p in Img_D.parameters():
            p.requires_grad = False
        for p in Tab_D.parameters():
            p.requires_grad = False
        for p in G.parameters():
            p.requires_grad = True
            
        for iters in range(g_iter):
            G.zero_grad()
            
            noise = torch.randn((batch_size, latent_dim)).to(device)
            condvec = data_sampler.sample_condvec(batch_size)

            if condvec is None: # unconditional
                c1, m1, col, opt = None, None, None, None

                fake_images, fake_images_small = G(noise, crop_idx=crop_idx, cond=None)

                y_fake_g = Img_D(fake_images, fake_images_small, crop_idx)
                g_loss = loss_f(y_fake_g, real_labels)                
            
            else: # conditional
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(device)
                m1 = torch.from_numpy(m1).to(device)

                fake_images, fake_images_small, fake_tab = G(noise, crop_idx=crop_idx, cond=c1)

                y_fake_g, y_fake_g_class = Img_D(fake_images, fake_images_small, crop_idx)
                g_loss = loss_f(y_fake_g, real_labels) + lambda_class * F.cross_entropy(y_fake_g_class, class_label)

            data_t = []
            st = 0

            for column_info in transformer.output_info_list:
                for span_info in column_info:
                    if span_info.activation_fn == 'tanh':
                        ed = st + span_info.dim
                        data_t.append(torch.tanh(fake_tab[:, st:ed]))
                        st = ed
                    elif span_info.activation_fn == 'softmax':
                        ed = st + span_info.dim
                        transformed = gumbel_softmax(fake_tab[:, st:ed], tau=0.2)
                        data_t.append(transformed)
                        st = ed
                    else:
                        raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

            fakeact = torch.cat(data_t, dim=1)
            
            if c1 is not None:
                y_fake = Tab_D(torch.cat([fakeact, c1], dim=1))
            else:
                y_fake = Tab_D(fakeact)
            
            if condvec is None:
                cross_entropy = 0
            else:
                '''
                    Compute the cross entropy loss on the fixed discrete column.
                '''
                loss = []
                st = 0
                st_c = 0

                for column_info in transformer.output_info_list:
                    for span_info in column_info:
                        if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                            st += span_info.dim
                        else:
                            ed = st + span_info.dim
                            ed_c = st_c + span_info.dim
                            tmp = F.cross_entropy(
                                fake_tab[:, st:ed],
                                torch.argmax(c1[:, st_c:ed_c], dim=1),
                                reduction='none'
                            )
                            loss.append(tmp)
                            st = ed
                            st_c = ed_c

                loss = torch.stack(loss, dim=1)

                cross_entropy = (loss * m1).sum() / fake_tab.size()[0]
            
            loss_g = -torch.mean(y_fake) + cross_entropy

            g_optim_tab.zero_grad(set_to_none=False)
            loss_g.backward(retain_graph=True)
            g_optim_tab.step()

            g_loss.backward()
            g_optim_img.step()
        

        ###############################################
        # Train Encoder (E^H)
        ###############################################
        
        for p in E.parameters():
            p.requires_grad = True
        for p in G.parameters():
            p.requires_grad = False
        
        E.zero_grad()
        
        z_hat = E(real_images_crop)
        x_hat = G(z_hat, crop_idx=None)
        
        e_loss = loss_mse(x_hat, real_images_crop)
        e_loss.backward()
        e_optim_img.step()


        ###############################################
        # Train Sub Encoder (E^G)
        ###############################################
        
        for p in Sub_E.parameters():
            p.requires_grad = True
        for p in E.parameters():
            p.requires_grad = False
        
        Sub_E.zero_grad()
        
        with torch.no_grad():
            z_hat_i_list = []

            # Process all sub-volume and concatenate
            for crop_idx_i in range(0,img_size,img_size//8):
                real_images_crop_i = real_images[:,:,crop_idx_i:crop_idx_i+img_size//8,:,:]
                z_hat_i = E(real_images_crop_i)
                z_hat_i_list.append(z_hat_i)
            z_hat = torch.cat(z_hat_i_list, dim=2).detach()
        sub_z_hat = Sub_E(z_hat)
        
        # Reconstruction
        if condvec is None: # unconditional
            sub_x_hat_rec, sub_x_hat_rec_small, _ = G(sub_z_hat, crop_idx=crop_idx)
        else: # conditional
            sub_x_hat_rec, sub_x_hat_rec_small, _ = G(sub_z_hat, crop_idx=crop_idx, cond=c1)
        
        sub_e_loss = (loss_mse(sub_x_hat_rec,real_images_crop) + loss_mse(sub_x_hat_rec_small,real_images_small))/2.

        sub_e_loss.backward()
        sub_e_optim_img.step()

        step_info = f'Step {id_+1}/{steps_per_epoch}'
        step_iterator.set_description(step_info)

    # # Logging
    # summary_writer.add_scalar('D', d_loss.item(), epoch)
    # summary_writer.add_scalar('D_real', d_real_loss.item(), epoch)
    # summary_writer.add_scalar('D_fake', d_fake_loss.item(), epoch)
    # summary_writer.add_scalar('G_fake', g_loss.item(), epoch)
    # summary_writer.add_scalar('E', e_loss.item(), epoch)
    # summary_writer.add_scalar('Sub_E', sub_e_loss.item(), epoch)

    epoch_info = f'Epoch {epoch+1}/{num_epochs}'
    epoch_iterator.set_description(epoch_info)

    torch.save({'model':G.state_dict(), 'optimizer':g_optim_img.state_dict()},'./checkpoint/G_img_epoch_'+str(epoch+1)+'.pth')
    torch.save({'model':G.state_dict(), 'optimizer':g_optim_tab.state_dict()},'./checkpoint/G_tab_epoch_'+str(epoch+1)+'.pth')
    torch.save({'model':E.state_dict(), 'optimizer':e_optim_img.state_dict()},'./checkpoint/E_epoch_'+str(epoch+1)+'.pth')
    torch.save({'model':Sub_E.state_dict(), 'optimizer':sub_e_optim_img.state_dict()},'./checkpoint/Sub_E_epoch_'+str(epoch+1)+'.pth')

