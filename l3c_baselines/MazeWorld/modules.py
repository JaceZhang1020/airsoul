import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, hidden_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(in_channel, hidden_size, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_size, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class Encoder(nn.Module):
    """
    Change [B*NT, C_in, 128, 128] to [B*NT, C_out]
    """
    def __init__(self, img_size, in_channel, out_channel, n_res_block):
        super().__init__()

        channel_b1 = out_channel // 16
        channel_b2 = out_channel // 8
        channel_b3 = out_channel // 4
        cur_size = img_size // 8
        fin_channel = cur_size * cur_size * channel_b3

        blocks = [
            nn.Conv2d(in_channel, channel_b1, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_b1, channel_b2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_b2, channel_b2, 3, padding=1),
        ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel_b2, channel_b2))

        blocks.extend([
            nn.Conv2d(channel_b2, channel_b2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_b2, channel_b3, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_b3, channel_b3, 3, padding=1),
        ])

        for i in range(n_res_block):
            blocks.append(ResBlock(channel_b3, channel_b3))

        blocks.extend([
            nn.Conv2d(channel_b3, channel_b3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_b3, channel_b3, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_b3, channel_b3, 3, padding=1),
        ])

        blocks.extend([
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(fin_channel, out_channel),
            nn.ReLU(),
        ])

        self.blocks = nn.Sequential(*blocks)
        self.hidden_size = out_channel

    def forward(self, input):
        return self.blocks(input)

class Decoder(nn.Module):
    """
    Change [B*NT, C_in] to [B*NT, C_out, 128, 128]
    """
    def __init__(
        self, 
        img_size,
        in_channel, 
        out_channel, 
        n_res_block, 
    ):
        super().__init__()

        channel_b1 = in_channel // 2
        self.ini_size = img_size // 8
        self.ini_channel = in_channel // 4
        ini_mapping = self.ini_size * self.ini_size * self.ini_channel

        self.input_mapping = nn.Sequential(nn.Linear(in_channel, ini_mapping), nn.ReLU())

        blocks = [nn.Conv2d(self.ini_channel, channel_b1, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel_b1, channel_b1))

        blocks.append(nn.ReLU())

        blocks.extend(
            [
                nn.ConvTranspose2d(channel_b1, channel_b1 // 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(channel_b1 // 2, channel_b1 // 4, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(channel_b1 // 4, channel_b1 // 8, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(channel_b1 // 8, out_channel, 3, padding=1),
                nn.Sigmoid(),
            ]
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs):
        img = self.input_mapping(inputs)
        img = img.view(-1, self.ini_channel, self.ini_size, self.ini_size)
        return self.blocks(img)

class MapDecoder(nn.Module):
    def __init__(
        self, 
        in_channel, 
        hidden,
        out_channel, 
        map_size,
    ):
        super().__init__()

        self.input_mapping = nn.Linear(in_channel, map_size * map_size * hidden)
        self.map_size = map_size
        self.hidden = hidden

        blocks = [
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channel),
        ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        out = self.input_mapping(input)
        out = out.view(-1, self.map_size, self.map_size, self.hidden)
        out = self.blocks(out).permute(0, 3, 1, 2)
        return out

class VAE(nn.Module):
    def __init__(
        self,
        hidden_size,
        encoder,
        decoder):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = encoder
        self.decoder = decoder
        self.layer_mean = nn.Linear(encoder.hidden_size, hidden_size)
        self.layer_var = nn.Linear(encoder.hidden_size, hidden_size)
    
    def forward(self, inputs):
        # input shape: [B, NT, C, W, H]
        nB, nT, nC, nW, nH = inputs.shape
        hidden = self.encoder(inputs.reshape(nB * nT, nC, nW, nH))
        z_exp = self.layer_mean(hidden)
        z_log_var = self.layer_var(hidden)
        return z_exp.reshape(nB, nT, self.hidden_size), z_log_var.reshape(nB, nT, self.hidden_size)

    def reconstruct(self, inputs):
        nB, nT, nC, nW, nH = inputs.shape
        z_exp, z_log_var = self.forward(inputs)
        epsilon = torch.randn_like(z_log_var).to(z_log_var.device)
        z = z_exp + torch.exp(z_log_var / 2) * epsilon
        outputs = self.decoder(z.reshape(nB * nT, self.hidden_size))
        outputs = outputs.reshape(nB, nT, nC, nW, nH)
        return outputs, z_exp, z_log_var

    def loss(self, inputs, _lambda=1.0e-5):
        outputs, z_exp, z_log_var = self.reconstruct(inputs)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - torch.square(z_exp) - torch.exp(z_log_var), axis=1))
        reconstruction_loss = F.mse_loss(outputs, inputs)

        return reconstruction_loss + _lambda * kl_loss

def mse_loss_mask(img_out, img_gt, mask = None):
    mse_loss = torch.mean(((img_out - img_gt / 255)) ** 2, dim=[2, 3, 4])
    if mask is not None:
        mse_loss = mse_loss * mask
        sum_mask = torch.sum(mask)
        sum_loss = torch.sum(mse_loss)
        mse_loss = sum_loss / sum_mask
    else:
        mse_loss = torch.mean(mse_loss)

    return mse_loss

def ce_loss_mask(act_out, act_gt, mask = None):
    act_logits = F.one_hot(act_gt, act_out.shape[-1])
    ce_loss = -torch.mean(torch.log(act_out) * act_logits, dim=-1)
    if mask is not None:
        ce_loss = ce_loss * mask
        sum_mask = torch.sum(mask)
        sum_loss = torch.sum(ce_loss)
        ce_loss = sum_loss / sum_mask
    else:
        ce_loss = torch.mean(ce_loss)

    return ce_loss

def img_pro(observations):
    return observations / 255

def img_post(observations):
    return observations * 255
