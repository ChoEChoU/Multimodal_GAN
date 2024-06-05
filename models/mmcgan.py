import torch
from torch import nn
from torch.nn import functional as F
from models.hagan.Model_HA_GAN_256 import Sub_Generator, Sub_Discriminator
from models.hagan.layers import SNConv3d, SNLinear
from models.ctgan.synthesizers.ctgan import Generator as Tab_Generator


class Bottleneck(nn.Module):
    def __init__(self, channel=64, out_dim=128):
        super(Bottleneck, self).__init__()
        _c = channel

        self.conv3d = nn.Conv3d(channel, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool3d = nn.MaxPool3d(kernel_size=2, stride=2)
        self.batchnorm = nn.BatchNorm3d(num_features=64)
        self.dense = nn.Linear(_c * (_c//2)**3, out_dim)
        
    def forward(self, x):
        x = self.conv3d(x)
        x = self.maxpool3d(x)
        x = self.batchnorm(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)

        return x


class Img_Discriminator(nn.Module):
    def __init__(self, num_class=0, channel=512, device='cuda'):
        super(Img_Discriminator, self).__init__()        
        self.channel = channel
        self.num_class = num_class
        self.device = device
        
        # D^H
        self.conv1 = SNConv3d(1, channel//32, kernel_size=4, stride=2, padding=1) # in:[32,256,256], out:[16,128,128]
        self.conv2 = SNConv3d(channel//32, channel//16, kernel_size=4, stride=2, padding=1) # out:[8,64,64,64]
        self.conv3 = SNConv3d(channel//16, channel//8, kernel_size=4, stride=2, padding=1) # out:[4,32,32,32]
        self.conv4 = SNConv3d(channel//8, channel//4, kernel_size=(2,4,4), stride=(2,2,2), padding=(0,1,1)) # out:[2,16,16,16]
        self.conv5 = SNConv3d(channel//4, channel//2, kernel_size=(2,4,4), stride=(2,2,2), padding=(0,1,1)) # out:[1,8,8,8]
        self.conv6 = SNConv3d(channel//2, channel, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)) # out:[1,4,4,4]
        self.conv7 = SNConv3d(channel, channel//4, kernel_size=(1,4,4), stride=1, padding=0) # out:[1,1,1,1]
        self.fc1 = SNLinear(channel//4+1, channel//8)
        self.fc2 = SNLinear(channel//8, 1)

        if num_class>0:
            self.fc2_class = SNLinear(channel//8, num_class)

        # D^L
        self.sub_D = Sub_Discriminator(num_class)

    def forward(self, h, h_small, crop_idx):
        h = F.leaky_relu(self.conv1(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv5(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv6(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv7(h), negative_slope=0.2).squeeze()
        h = torch.cat([h, (crop_idx / 224. * torch.ones((h.size(0), 1))).to(self.device)], 1) # 256*7/8
        h = F.leaky_relu(self.fc1(h), negative_slope=0.2)
        h_logit = self.fc2(h)

        if self.num_class>0:
            h_class_logit = self.fc2_class(h)
            h_small_logit, h_small_class_logit = self.sub_D(h_small)

            return (h_logit + h_small_logit)/2., (h_class_logit + h_small_class_logit)/2.
        else:
            h_small_logit = self.sub_D(h_small)

            return (h_logit + h_small_logit)/2.


class Generator(nn.Module):
    def __init__(self, embedding_dim, generator_dim, data_dim, condition_dim,
                 mode="train", latent_dim=1024, channel=32):
        super(Generator, self).__init__()
        _c = channel

        self.emb_dim = embedding_dim
        self.gen_dim = generator_dim
        self.data_dim = data_dim
        self.cond_dim = condition_dim

        self.mode = mode
        self.relu = nn.ReLU()

        # G^A and G^H
        self.fc1 = nn.Linear(latent_dim + condition_dim, 4*4*4*_c*16)

        self.tp_conv1 = nn.Conv3d(_c*16, _c*16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.GroupNorm(8, _c*16)

        self.tp_conv2 = nn.Conv3d(_c*16, _c*16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(8, _c*16)

        self.tp_conv3 = nn.Conv3d(_c*16, _c*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.GroupNorm(8, _c*8)

        self.tp_conv4 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.GroupNorm(8, _c*4)

        self.tp_conv5 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.GroupNorm(8, _c*2)

        self.tp_conv6 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.GroupNorm(8, _c)

        self.tp_conv7 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=True)

        # G^L
        self.sub_G = Sub_Generator(channel=_c//2)

        # G^Tab
        self.bottleneck = Bottleneck(channel=64, out_dim=128)
        self.tab_G = Tab_Generator(self.emb_dim + self.cond_dim, self.gen_dim, self.data_dim)
 
    def forward(self, h, crop_idx=None, cond=None):

        # Generate from random noise
        if crop_idx != None or self.mode=='eval':
            if cond != None:
                h = torch.cat((h, cond), dim=1)

            h = self.fc1(h)

            h = h.view(-1,512,4,4,4)
            h = self.tp_conv1(h)
            h = self.relu(self.bn1(h))

            h = F.interpolate(h,scale_factor = 2)
            h = self.tp_conv2(h)
            h = self.relu(self.bn2(h))

            h = F.interpolate(h,scale_factor = 2)
            h = self.tp_conv3(h)
            h = self.relu(self.bn3(h))

            h = F.interpolate(h,scale_factor = 2)
            h = self.tp_conv4(h)
            h = self.relu(self.bn4(h))

            h = F.interpolate(h,scale_factor = 2)
            h = self.tp_conv5(h)
            h_latent = self.relu(self.bn5(h)) # (64, 64, 64), channel:128

            if self.mode == "train":
                h_small = self.sub_G(h_latent)
                h = h_latent[:,:,crop_idx//4:crop_idx//4+8,:,:] # Crop, out: (8, 64, 64)
            else:
                h = h_latent
            
            h_bn = self.bottleneck(h_latent)
            h_bn = torch.cat((h_bn, cond), dim=1)
            h_tab = self.tab_G(h_bn)

        # Generate from latent feature
        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv6(h)
        h = self.relu(self.bn6(h)) # (128, 128, 128)

        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv7(h)

        h = torch.tanh(h) # (256, 256, 256)

        if crop_idx != None and self.mode == "train":
            return h, h_small, h_tab
        elif self.mode == 'eval':
            return h, h_tab
        
        return h


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    """Deals with the instability of the gumbel_softmax for older versions of torch.

    For more details about the issue:
    https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

    Args:
        logits […, num_features]:
            Unnormalized log probabilities
        tau:
            Non-negative scalar temperature
        hard (bool):
            If True, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
        dim (int):
            A dimension along which softmax will be computed. Default: -1.

    Returns:
        Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
    """
    for _ in range(10):
        transformed = F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
        if not torch.isnan(transformed).any():
            return transformed

    raise ValueError('gumbel_softmax returning NaN.')


def inf_train_gen(data_loader):
    while True:
        for _, batch in enumerate(data_loader):
            yield batch


def sample_tab(n, transformer, data_sampler, batch_size, latent_dim, device, generator, condition_column=None, condition_value=None):
    """Sample data similar to the training data.

    Choosing a condition_column and condition_value will increase the probability of the
    discrete condition_value happening in the condition_column.

    Args:
        n (int):
            Number of rows to sample.
        condition_column (string):
            Name of a discrete column.
        condition_value (string):
            Name of the category in the condition_column which we wish to increase the
            probability of happening.

    Returns:
        numpy.ndarray or pandas.DataFrame
    """
    if condition_column is not None and condition_value is not None:
        condition_info = transformer.convert_column_name_value_to_id(
            condition_column, condition_value)
        global_condition_vec = data_sampler.generate_cond_from_condition_column_info(
            condition_info, batch_size)
    else:
        global_condition_vec = None

    steps = n // batch_size + 1
    data = []
    for _ in range(steps):
        noise = torch.randn((batch_size, latent_dim)).to(device)

        if global_condition_vec is not None:
            condvec = global_condition_vec.copy()
        else:
            condvec = data_sampler.sample_original_condvec(batch_size)

        if condvec is None:
            pass
        else:
            condvec = torch.from_numpy(condvec).to(device)

        _, _, fake = generator(noise, crop_idx=crop_idx, cond=condvec)
    
        data_t = []
        st = 0
        for column_info in transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(fake[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = gumbel_softmax(fake[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        fakeact = torch.cat(data_t, dim=1)
        data.append(fakeact.detach().cpu().numpy())

    data = np.concatenate(data, axis=0)
    data = data[:n]

    return transformer.inverse_transform(data)





