import torch
import torch.nn.functional as F
from scipy import io
import numpy as np
from matplotlib import pyplot as plt
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# make a grid of frequencies

x_rfft2 = torch.fft.rfft2(torch.ones(512,512).cuda())
# x_rfft2 = torch.fft.rfft2(torch.ones(256,256).cuda())
# x_rfft2 = torch.fft.rfft2(torch.ones(128,128).cuda())

df = 1/x_rfft2.shape[0]
f1 = torch.arange((x_rfft2.shape[0]-1)//2).cuda()*df
f1 = torch.cat((f1, torch.arange(-(x_rfft2.shape[0]+1)//2, 0).cuda()*df))
f2 = torch.arange(x_rfft2.shape[1]).cuda()*df
f1, f2 = torch.meshgrid(f1, f2)

# compute the MTF

f_co_1 = 0.02
f_co_2 = 0.05
f_co_3 = 0.15
f_co_4 = 0.35

f_c_task = 0.3
f_co_task = 0.02

H_1 = torch.exp(-(f1**2+f2**2)/(2*f_co_1**2))
H_2 = torch.exp(-(f1**2+f2**2)/(2*f_co_2**2))
H_3 = torch.exp(-(f1**2+f2**2)/(2*f_co_3**2))
H_task = torch.exp(-(torch.sqrt(f1**2+f2**2) - f_c_task)**2/(2*f_co_task**2))

H_NearZero = H_1
H_Low = H_2 - H_1
H_Mid = H_3 - H_2
H_High = 1 - H_3
# H_task = H_task - H_3

max_noise_power = 0.05

H_NearZero = H_NearZero.to(device)
H_Low = H_Low.to(device)
H_Mid = H_Mid.to(device)
H_High = H_High.to(device)
H_task = H_task.to(device)


def make_basis_filters(shape):

    x_rfft2 = torch.fft.rfft2(torch.ones(shape[-2],shape[-1]).cuda())
    df = 1/x_rfft2.shape[0]
    f1 = torch.arange((x_rfft2.shape[0]-1)//2).cuda()*df
    f1 = torch.cat((f1, torch.arange(-(x_rfft2.shape[0]+1)//2, 0).cuda()*df))
    f2 = torch.arange(x_rfft2.shape[1]).cuda()*df
    f1, f2 = torch.meshgrid(f1, f2)

    # compute the MTF
    f_co_1 = 0.05
    f_co_2 = 0.15
    f_co_3 = 0.30

    f_c_task = 0.25
    f_co_task = 0.05

    H_1 = torch.exp(-(f1**2+f2**2)/(2*f_co_1**2))
    H_2 = torch.exp(-(f1**2+f2**2)/(2*f_co_2**2))
    H_3 = torch.exp(-(f1**2+f2**2)/(2*f_co_3**2))
    H_task = torch.exp(-(torch.sqrt(f1**2+f2**2) - f_c_task)**2/(2*f_co_task**2))

    H_NearZero = H_1
    H_Low = H_2 - H_1
    H_Mid = H_3 - H_2
    H_High = 1 - H_3

    max_noise_power = 0.05

    H_NearZero = H_NearZero.to(device)
    H_Low = H_Low.to(device)
    H_Mid = H_Mid.to(device)
    H_High = H_High.to(device)
    H_task = H_task.to(device)

    return H_NearZero, H_Low, H_Mid, H_High, H_task


def MTF_kernel(alpha1, alpha2, alpha3, alpha4, zeroMean=False, shape=[512,512]):

    H_NearZero, H_Low, H_Mid, H_High, H_task = make_basis_filters(shape)

    H = H_NearZero + alpha1*H_Low + alpha2*H_Mid + alpha3*H_High + alpha4*H_task
    
    if zeroMean:
        H = H - H_NearZero
    H[H>1.0] = 1.0
    H[H<0.0] = 0.0
    return H

def NPS_kernel(alpha5, alpha6, alpha7, alpha8, zeroMean=False, shape=[512,512]):

    H_NearZero, H_Low, H_Mid, H_High, H_task = make_basis_filters(shape)

    H = 0.001*H_NearZero + alpha5*H_Low + alpha6*H_Mid + alpha7*H_High + alpha8*H_task
    if zeroMean:
        H = H - 0.001*H_NearZero
    H = H*max_noise_power
    H[H>1.0] = 1.0
    H[H<1e-8] = 1e-8
    return H

def apply_MTF(x, alpha1, alpha2, alpha3, alpha4,zeroMean=False, pad=False):
    x_rfft2 = torch.fft.rfft2(x)
    H = MTF_kernel(alpha1, alpha2, alpha3, alpha4, zeroMean=zeroMean, shape=x.shape)
    x_rfft2 = x_rfft2*H
    x_out = torch.fft.irfft2(x_rfft2)
    return x_out

def apply_NPS(x, alpha5, alpha6, alpha7, alpha8, zeroMean=False):
    z = torch.randn(x.shape).cuda()
    z_rfft2 = torch.fft.rfft2(z)
    H = NPS_kernel(alpha5, alpha6, alpha7, alpha8, zeroMean=zeroMean, shape=x.shape)
    z_rfft2 = z_rfft2*torch.sqrt(H)
    z = torch.fft.irfft2(z_rfft2)
    return x + z










N_batch=16

# make a neural network model with one input and one output
import torch.nn as nn
import torch.nn.functional as F

class Block(torch.nn.Module):
    def __init__(self, in_ch, out_ch, pass_ch=18):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 3)
        self.conv2 = torch.nn.ConvTranspose2d(out_ch, out_ch, 3)
        self.conv3 = torch.nn.Conv2d(out_ch, out_ch, 3)
        self.conv4 = torch.nn.ConvTranspose2d(out_ch, out_ch-pass_ch, 3)
        self.relu  = torch.nn.ReLU()
        self.pass_ch = pass_ch
        # self.batch_norm = torch.nn.BatchNorm2d(out_ch-pass_ch)
    def forward(self, x):
        out = x
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        # out = self.batch_norm(out)
        out = torch.cat((x[...,0:self.pass_ch,:,:], out), dim=-3)
        return out

# class Block(torch.nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 3)
#         self.relu  = torch.nn.ReLU()
#         self.conv2 = torch.nn.ConvTranspose2d(out_ch, out_ch, 3)
#         self.batch_norm = torch.nn.BatchNorm2d(out_ch)
#     def forward(self, x):
#         out = self.relu(self.conv2(self.relu(self.conv1(x))))
#         out = self.batch_norm(out)
#         return out

class Encoder(torch.nn.Module):
    def __init__(self, chs=(2,32,32,32,32),pass_ch=5):
        super().__init__()
        self.enc_blocks = torch.nn.ModuleList([Block(chs[i], chs[i+1],pass_ch=pass_ch) for i in range(len(chs)-1)])
        self.pool       = torch.nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        ftrs[-1] = torch.nn.Dropout(p=0.5)(ftrs[-1])
        return ftrs

class Decoder(torch.nn.Module):
    def __init__(self, chs=( 32, 32, 32, 32),pass_ch=5):
        super().__init__()
        self.chs         = chs
        self.upconvs    = torch.nn.ModuleList([torch.nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = torch.nn.ModuleList([Block(chs[i], chs[i+1], pass_ch=pass_ch) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            x        = torch.cat([encoder_features[i], x], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
class UNet(torch.nn.Module):
    def __init__(self, enc_chs=(5,32,64,128,256), dec_chs=( 256, 128, 64, 32,1),pass_ch=5):
        super().__init__()
        self.encoder     = Encoder(enc_chs,pass_ch=pass_ch)
        self.decoder     = Decoder(dec_chs[:-1],pass_ch=pass_ch)
        self.head        = torch.nn.ConvTranspose2d(dec_chs[-2], dec_chs[-1], 1)

    def forward(self, x):
        # y = x.permute(0,3,1,2) # channels last -> channels first
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        # out = out.permute(0,2,3,1) # channels first -> channels last
        # return x[...,0:2] + out
        # out = 1*torch.sigmoid(out) + .0001
        # out = torch.relu(out)
        # out = x[:,0:1] + out
        return out


# shallow cnn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.ConvTranspose2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.ConvTranspose2d(128, 256, 3, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1)
        self.conv6 = nn.ConvTranspose2d(128, 1, 3, 1)
    def forward(self, x):
        x_in = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        # x = self.conv6(x)
        x = F.relu(self.conv6(x))
        return x


def log_likelihood_loss(mu_true, mu_pred, alpha5, alpha6, alpha7, alpha8):
    H_NPS = NPS_kernel(alpha5, alpha6, alpha7, alpha8)
    mu_true_rfft2 = torch.fft.rfft2(mu_true)
    mu_pred_rfft2 = torch.fft.rfft2(mu_pred)
    return torch.mean(torch.abs(mu_true_rfft2 - mu_pred_rfft2)**2 / H_NPS)





def compute_kl_divergence_parameters(H_MTF_input, H_MTF_target, H_NPS_input, H_NPS_target):

    H_LSI = H_MTF_input/H_MTF_target
    H_epsilon = H_NPS_input - H_LSI*H_NPS_target*H_LSI
    H_delta = (1/H_LSI)*H_epsilon*(1/H_LSI)
    H_NPS_posterior = H_NPS_target*H_delta/(H_NPS_target + H_delta)

    return H_LSI, H_epsilon, H_delta, H_NPS_posterior



def kl_divergence_loss(x, x_input, mu_pred, 
                       alpha1, alpha2, alpha3, alpha4, 
                       alpha5, alpha6, alpha7, alpha8,
                       target_alpha1, target_alpha2, target_alpha3, target_alpha4,
                       target_alpha5, target_alpha6, target_alpha7, target_alpha8):

    H_MTF_input = MTF_kernel(alpha1, alpha2, alpha3, alpha4, shape=x.shape)
    H_NPS_input = NPS_kernel(alpha5, alpha6, alpha7, alpha8, shape=x.shape)
    H_MTF_target = MTF_kernel(target_alpha1, target_alpha2, target_alpha3, target_alpha4, shape=x.shape)
    H_NPS_target = NPS_kernel(target_alpha5, target_alpha6, target_alpha7, target_alpha8, shape=x.shape)

    H_LSI, H_epsilon, H_delta, H_NPS_posterior = compute_kl_divergence_parameters(H_MTF_input, H_MTF_target, H_NPS_input, H_NPS_target)

    x_rfft2 = torch.fft.rfft2(x)
    x_input_rfft2 = torch.fft.rfft2(x_input)
    mu_pred_rfft2 = torch.fft.rfft2(mu_pred)

    mu_posterior_rfft2 = 0
    # likelihood term
    mu_posterior_rfft2 = mu_posterior_rfft2 + (H_NPS_target/(H_NPS_target + H_delta))*(x_input_rfft2/H_LSI)
    # prior term
    mu_posterior_rfft2 = mu_posterior_rfft2 + (H_delta/(H_NPS_target + H_delta))* (x_rfft2*H_MTF_target)
    mu_posterior = torch.fft.irfft2(mu_posterior_rfft2)

    
    # have to do this to avoid divide by zero
    H_NPS_posterior[H_NPS_posterior<1e-6] = 1e-6
    # H_NPS_posterior[H_NPS_posterior<1.0] = 1.0
    

    loss = torch.mean((torch.abs(mu_posterior_rfft2 - mu_pred_rfft2)/torch.sqrt(H_NPS_posterior))**2.0)

    return loss, mu_posterior

    



def load_training_data(N, quiet=False, max=None, min = None, seed=None):
    training_data = torch.zeros((N, 1, 512,512))
    # training_data = torch.zeros((N, 1, 256,256))
    # training_data = torch.zeros((N, 1, 128,128))

    if max is None:
        max = N
    if min is None:
        min = 1

    if seed is not None:
        random.seed(seed)

    random_list = random.sample(range(min, max), N)
    random_list.sort()

    jPatient = 0
    for iPatient in random_list:
        if not quiet:
            print("Loading patient ", iPatient, " of ", max)
        x = torch.tensor(io.loadmat('../20220321_LIDC/data/LIDC/x/x_'+str(iPatient).zfill(4)+'.mat')['x']).cuda()
        training_data[jPatient,0,:,:] = x
        # training_data[iPatient,0,:,:] = x[::2,::2]
        jPatient = jPatient + 1

    return training_data




def load_training_data_patch(N, quiet=False, max=None, min = None, seed=None, patches_per_patient=8, patch_size=64):
    training_data = torch.zeros((N, 1, patch_size, patch_size))

    if max is None:
        max = N
    if min is None:
        min = 1

    if seed is not None:
        random.seed(seed)

    random_list = random.sample(range(min, max), int(np.ceil(N/patches_per_patient)))
    random_list.sort()

    jPatient = 0
    for iPatient in range(N):
        if iPatient % patches_per_patient == 0:
            kPatient = random_list[jPatient]
            if not quiet:
                print("Loading patient ", kPatient, " of ", max)
            x = torch.tensor(io.loadmat('../20220321_LIDC/data/LIDC/x/x_'+str(kPatient).zfill(4)+'.mat')['x']).cuda()
            jPatient = jPatient + 1
    
            iRow = random.randint(64, 512-64-patch_size)
            iCol = random.randint(64, 512-64-patch_size)
            
            training_data[iPatient,0,:,:] = x[iRow:iRow+patch_size,iCol:iCol+patch_size]

    training_data=training_data[torch.randperm(N)]

    return training_data


def random_MTF_NPS_parameters(device=device):

    valid_flag = False

    # this is actually dt/T
    dt = 1/50

    # this is actually t/T
    t = torch.rand(1).to(device)*(1-dt) + dt

    alpha1 = torch.ones([1], device=device)*(2.5 - 2.0*((t + 0.25)))
    alpha2 = torch.ones([1], device=device)*(2.5 - 2.0*((t + 0.50)))
    alpha3 = torch.ones([1], device=device)*(2.5 - 2.0*((t + 0.75)))
    alpha4 = torch.ones([1], device=device)*(0.1 - 0.1*((t + 0.00)))
    alpha5 = torch.ones([1], device=device)*(0.0 + 0.8*((t + 0.00)))*0.1
    alpha6 = torch.ones([1], device=device)*(0.0 + 1.0*((t + 0.00)))*0.1
    alpha7 = torch.ones([1], device=device)*(0.0 + 0.5*((t + 0.00)))*0.1
    alpha8 = torch.ones([1], device=device)*(-0.1 + 0.1*((t + 0.00)))

    t = t - dt

    target_alpha1 = torch.ones([1], device=device)*(2.5 - 2.0*((t + 0.25)))
    target_alpha2 = torch.ones([1], device=device)*(2.5 - 2.0*((t + 0.50)))
    target_alpha3 = torch.ones([1], device=device)*(2.5 - 2.0*((t + 0.75)))
    target_alpha4 = torch.ones([1], device=device)*(0.1 - 0.1*((t + 0.00)))
    target_alpha5 = torch.ones([1], device=device)*(0.0 + 0.8*((t + 0.00)))*0.1
    target_alpha6 = torch.ones([1], device=device)*(0.0 + 1.0*((t + 0.00)))*0.1
    target_alpha7 = torch.ones([1], device=device)*(0.0 + 0.5*((t + 0.00)))*0.1
    target_alpha8 = torch.ones([1], device=device)*(-0.1 + 0.1*((t + 0.00)))

    taskiness = torch.rand(1).to(device)*1.2
    taskiness = 0.0


    alpha4 = taskiness*alpha4
    alpha8 = taskiness*alpha8
    target_alpha4 = taskiness*target_alpha4
    target_alpha8 = taskiness*target_alpha8

    alpha1 = torch.maximum(alpha1, 0.0*torch.ones([1], device=device))
    alpha2 = torch.maximum(alpha2, 0.0*torch.ones([1], device=device))
    alpha3 = torch.maximum(alpha3, 0.0*torch.ones([1], device=device))
    # alpha4 = torch.maximum(alpha4, 0.0*torch.ones([1], device=device))
    alpha5 = torch.maximum(alpha5, 0.0*torch.ones([1], device=device))
    alpha6 = torch.maximum(alpha6, 0.0*torch.ones([1], device=device))
    alpha7 = torch.maximum(alpha7, 0.0*torch.ones([1], device=device))
    # alpha8 = torch.maximum(alpha8, 0.0*torch.ones([1], device=device))

    alpha1 = torch.minimum(alpha1, 1.0*torch.ones([1], device=device))
    alpha2 = torch.minimum(alpha2, 1.0*torch.ones([1], device=device))
    alpha3 = torch.minimum(alpha3, 1.0*torch.ones([1], device=device))
    # alpha4 = torch.minimum(alpha4, 1.0*torch.ones([1], device=device))
    alpha5 = torch.minimum(alpha5, 1.0*torch.ones([1], device=device))
    alpha6 = torch.minimum(alpha6, 1.0*torch.ones([1], device=device))
    alpha7 = torch.minimum(alpha7, 1.0*torch.ones([1], device=device))
    # alpha8 = torch.minimum(alpha8, 1.0*torch.ones([1], device=device))

    target_alpha1 = torch.maximum(target_alpha1, 0.0*torch.ones([1], device=device))
    target_alpha2 = torch.maximum(target_alpha2, 0.0*torch.ones([1], device=device))
    target_alpha3 = torch.maximum(target_alpha3, 0.0*torch.ones([1], device=device))
    # target_alpha4 = torch.maximum(target_alpha4, 0.0*torch.ones([1], device=device))
    target_alpha5 = torch.maximum(target_alpha5, 0.0*torch.ones([1], device=device))
    target_alpha6 = torch.maximum(target_alpha6, 0.0*torch.ones([1], device=device))
    target_alpha7 = torch.maximum(target_alpha7, 0.0*torch.ones([1], device=device))
    # target_alpha8 = torch.maximum(target_alpha8, 0.0*torch.ones([1], device=device))

    target_alpha1 = torch.minimum(target_alpha1, 1.0*torch.ones([1], device=device))
    target_alpha2 = torch.minimum(target_alpha2, 1.0*torch.ones([1], device=device))
    target_alpha3 = torch.minimum(target_alpha3, 1.0*torch.ones([1], device=device))
    # target_alpha4 = torch.minimum(target_alpha4, 1.0*torch.ones([1], device=device))
    target_alpha5 = torch.minimum(target_alpha5, 1.0*torch.ones([1], device=device))
    target_alpha6 = torch.minimum(target_alpha6, 1.0*torch.ones([1], device=device))
    target_alpha7 = torch.minimum(target_alpha7, 1.0*torch.ones([1], device=device))
    # target_alpha8 = torch.minimum(target_alpha8, 1.0*torch.ones([1], device=device))

    return alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8, target_alpha1, target_alpha2, target_alpha3, target_alpha4, target_alpha5, target_alpha6, target_alpha7, target_alpha8

# def random_target_MTF_NPS_parameters(alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8, device=None):

#     valid_flag = False

#     while not valid_flag:
        
#         t = (-40*torch.log(alpha1) - 1)*torch.ones(1,device=device)

#         target_alpha1 = torch.exp(-t[0]/40)
#         target_alpha2 = torch.exp(-t[0]/30)
#         target_alpha3 = torch.exp(-t[0]/20)
#         target_alpha4 = 0*target_alpha3
#         target_alpha5 = (1/np.exp(100/40))*torch.exp(t[0]/40)
#         target_alpha6 = (1/np.exp(100/30))*torch.exp(t[0]/30)
#         target_alpha7 = (1/np.exp(100/20))*torch.exp(t[0]/20)
#         target_alpha8 = 0*target_alpha7



#         # target_alpha1 = alpha1 + .05*(1 - alpha1)
#         # target_alpha2 = alpha2*target_alpha1/alpha1
#         # target_alpha3 = alpha3*target_alpha2/alpha2
#         # target_alpha4 = 0*target_alpha3
        
#         # target_alpha5 = alpha5 + .05*(0-alpha5)
#         # target_alpha6 = alpha6*target_alpha5/alpha5
#         # target_alpha7 = alpha7*target_alpha6/alpha6
#         # target_alpha8 = 0*target_alpha6

#         if device is not None:
#             target_alpha1 = target_alpha1.to(device)
#             target_alpha2 = target_alpha2.to(device)
#             target_alpha3 = target_alpha3.to(device)
#             target_alpha4 = target_alpha4.to(device)
#             target_alpha5 = target_alpha5.to(device)
#             target_alpha6 = target_alpha6.to(device)
#             target_alpha7 = target_alpha7.to(device)
#             target_alpha8 = target_alpha8.to(device)

        
#         H_MTF = MTF_kernel(target_alpha1, target_alpha2, target_alpha3, target_alpha4)

#         H_NPS = NPS_kernel(target_alpha5, target_alpha6, target_alpha7, target_alpha8)

#         if torch.all(H_MTF > 0) and torch.all(H_NPS > 0):
#             valid_flag = True
        
#     return target_alpha1, target_alpha2, target_alpha3, target_alpha4, target_alpha5, target_alpha6, target_alpha7, target_alpha8















def initialize_plot(device=device):

    x = torch.zeros([1,1,x_rfft2.shape[0], x_rfft2.shape[0]]).to(device)

    fig = plt.figure(figsize=(16,8))
    # make 1,3 subplot
    ax1 = fig.add_subplot(1,3,1)
    im1 = ax1.imshow(x[0,0,:,:].cpu().numpy(), vmin=.2, vmax=.5, cmap='gray')
    # fig.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(1,3,2)
    im2 = ax2.imshow(x[0,0,:,:].cpu().numpy(), vmin=.2, vmax=.5, cmap='gray')
    # fig.colorbar(im2, ax=ax2)

    
    ax3 = fig.add_subplot(2,3,3)
    ln_MTF = ax3.plot(x[0,0,0,:].cpu().numpy(), color =[0.34,0.42,0.40], label='current MTF')
    ln_MTF_target = ax3.plot(x[0,0,0,:].cpu().numpy(), '--', color =[0.68,0.85,0.80], label='target MTF')
    plt.legend()
    ax4 = fig.add_subplot(2,3,6)
    ln_NPS = ax4.plot(x[0,0,0,:].cpu().numpy(), color=[0.133, 0.545, 0.133], label='current NPS')
    ln_NPS_target = ax4.plot(x[0,0,0,:].cpu().numpy(),'--', color=[0.563, 0.930, 0.563], label='target NPS')
    plt.legend()
    return fig, ax1, ax2, ax3, ax4, im1, im2, ln_MTF, ln_MTF_target, ln_NPS, ln_NPS_target

def update_plot(x1, x2, H_MTF, H_MTF_target,H_NPS, H_NPS_target,  fig, ax1, ax2, ax3, ax4, im1, im2, ln_MTF, ln_MTF_target, ln_NPS, ln_NPS_target, device=device):
    
    # fft shift H_MTF which is in rfft format on dimension 0, H_MTF has shape [256,127]
    H_MTF = torch.roll(H_MTF, H_MTF.shape[0]//2, dims=0)
    H_MTF_target = torch.roll(H_MTF_target, H_MTF.shape[0]//2, dims=0)
    H_NPS = torch.roll(H_NPS, H_MTF.shape[0]//2, dims=0)
    H_NPS_target = torch.roll(H_NPS_target, H_MTF.shape[0]//2, dims=0)


    im1.set_data(x1[0,0,:,:].detach().cpu().numpy())
    # ax2.set_title('x_meas, t = ' + str(t[0].item()))


    im2.set_data(x2[0,0,:,:].detach().cpu().numpy())
    # ax2.set_title('x_hat, t = ' + str(t[0].item()))

    ln_MTF[0].set_xdata(np.linspace(-0.5,0.5,x_rfft2.shape[0]))
    ln_MTF[0].set_ydata(H_MTF[:,0].detach().cpu().numpy())
    ln_MTF_target[0].set_xdata(np.linspace(-0.5,0.5,x_rfft2.shape[0]))
    ln_MTF_target[0].set_ydata(H_MTF_target[:,0].detach().cpu().numpy())
    ax3.set_title('Modulation Transfer Function')
    ax3.set_xlim([-0.5,0.5])
    ax3.set_ylim([0,1.1])
    ax3.set_xlabel('Spatial Frequency')
    ax3.set_ylabel('Modulation Transfer Function')

    ln_NPS[0].set_xdata(np.linspace(-0.5,0.5,x_rfft2.shape[0]))
    ln_NPS[0].set_ydata(H_NPS[:,0].detach().cpu().numpy())
    ln_NPS_target[0].set_xdata(np.linspace(-0.5,0.5,x_rfft2.shape[0]))
    ln_NPS_target[0].set_ydata(H_NPS_target[:,0].detach().cpu().numpy())
    ax4.set_title('Noise Power Spectum')
    ax4.set_xlim([-0.5,0.5])
    ax4.set_ylim([0,.08])
    ax4.set_xlabel('Spatial Frequency')
    ax4.set_ylabel('Noise Power Spectral Density')

    










def initialize_plot_2(device=device):

    x = torch.zeros([1,1,x_rfft2.shape[0], x_rfft2.shape[0]]).to(device)

    fig = plt.figure(figsize=(14,6))
    # make 1,3 subplot
    ax1 = fig.add_subplot(2,3,1)
    im1 = ax1.imshow(x[0,0,:,:].cpu().numpy(), vmin=.2, vmax=.5, cmap='gray')
    # fig.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(2,3,2)
    im2 = ax2.imshow(x[0,0,:,:].cpu().numpy(), vmin=.2, vmax=.5, cmap='gray')
    # fig.colorbar(im2, ax=ax2)

    # make 1,3 subplot
    ax4 = fig.add_subplot(2,3,4)
    im4 = ax4.imshow(x[0,0,:,:].cpu().numpy(), vmin=.2, vmax=.5, cmap='gray')
    # fig.colorbar(im1, ax=ax1)

    ax5 = fig.add_subplot(2,3,5)
    im5 = ax5.imshow(x[0,0,:,:].cpu().numpy(), vmin=.2, vmax=.5, cmap='gray')
    # fig.colorbar(im2, ax=ax2)




    
    ax3 = fig.add_subplot(2,3,3)
    ln_MTF = ax3.plot(x[0,0,0,:].cpu().numpy(), color =[0.34,0.42,0.40], label='current MTF')
    ln_MTF_target = ax3.plot(x[0,0,0,:].cpu().numpy(), '--', color =[0.68,0.85,0.80], label='target MTF')
    plt.legend()


    ax6 = fig.add_subplot(2,3,6)
    ln_NPS = ax6.plot(x[0,0,0,:].cpu().numpy(), color=[0.133, 0.545, 0.133], label='current NPS')
    ln_NPS_target = ax6.plot(x[0,0,0,:].cpu().numpy(),'--', color=[0.563, 0.930, 0.563], label='target NPS')




    plt.legend()
    return fig, ax1, ax2, ax3, ax4, ax5, ax6,  im1, im2, im4, im5, ln_MTF, ln_MTF_target, ln_NPS, ln_NPS_target

def update_plot_2(x1, x2, x4, x5, H_MTF, H_MTF_target,H_NPS, H_NPS_target,  fig, ax1, ax2, ax3, ax4, ax5, ax6, im1, im2, im4, im5, ln_MTF, ln_MTF_target, ln_NPS, ln_NPS_target, device=device):
    
    # fft shift H_MTF which is in rfft format on dimension 0, H_MTF has shape [256,127]
    H_MTF = torch.roll(H_MTF, H_MTF.shape[0]//2, dims=0)
    H_MTF_target = torch.roll(H_MTF_target, H_MTF.shape[0]//2, dims=0)
    H_NPS = torch.roll(H_NPS, H_MTF.shape[0]//2, dims=0)
    H_NPS_target = torch.roll(H_NPS_target, H_MTF.shape[0]//2, dims=0)

    im1.set_data(x1[0,0,:,:].detach().cpu().numpy())

    im2.set_data(x2[0,0,:,:].detach().cpu().numpy())

    im4.set_data(x4[0,0,:,:].detach().cpu().numpy())
    
    im5.set_data(x5[0,0,:,:].detach().cpu().numpy())

    ln_MTF[0].set_xdata(np.linspace(-0.5,0.5,x_rfft2.shape[0]))
    ln_MTF[0].set_ydata(H_MTF[:,0].detach().cpu().numpy())
    ln_MTF_target[0].set_xdata(np.linspace(-0.5,0.5,x_rfft2.shape[0]))
    ln_MTF_target[0].set_ydata(H_MTF_target[:,0].detach().cpu().numpy())
    ax3.set_title('Modulation Transfer Function')
    ax3.set_xlim([-0.5,0.5])
    ax3.set_ylim([0,1.1])
    # ax3.set_xlabel('Spatial Frequency')
    ax3.set_ylabel('Modulation Transfer Function')

    ln_NPS[0].set_xdata(np.linspace(-0.5,0.5,x_rfft2.shape[0]))
    ln_NPS[0].set_ydata(H_NPS[:,0].detach().cpu().numpy())
    ln_NPS_target[0].set_xdata(np.linspace(-0.5,0.5,x_rfft2.shape[0]))
    ln_NPS_target[0].set_ydata(H_NPS_target[:,0].detach().cpu().numpy())
    ax6.set_title('Noise Power Spectum')
    ax6.set_xlim([-0.5,0.5])
    ax6.set_ylim([0,.08])
    ax6.set_xlabel('Spatial Frequency')
    ax6.set_ylabel('Noise Power Spectral Density')

    






