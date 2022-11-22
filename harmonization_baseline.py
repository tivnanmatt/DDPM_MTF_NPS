import torch
import numpy as np
from matplotlib import pyplot as plt
import wandb

from utils import UNet
from utils import load_training_data, load_training_data_patch
from utils import kl_divergence_loss, compute_kl_divergence_parameters
from utils import initialize_plot_2, update_plot_2
from utils import MTF_kernel, NPS_kernel
from utils import apply_MTF, apply_NPS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

user = "tivnanmatt"
project = "harmonization_baseline"

wandb.init(entity=user, project=project)

learning_rate = 1e-5
nEpoch = 1000000
nPatientsPerBatch = 32
nPatchesPerPatient = 4
nBatchesPerEpoch_Training = 64
nBatchesPerEpoch_Validation = 32
patch_size = 128
nEpochsPerVideo = 10
N = 16
loadModelsFlag = True

enc_chs=(   2,  32, 64, 128, 256)
dec_chs=( 256, 128, 64,  32,   1)

# enc_chs=(  3, 32, 32, 32, 32)
# dec_chs=( 32, 32, 32, 32,  1)

config = {'learning_rate:': learning_rate, 
            'nEpoch': nEpoch,
            'nPatientsPerBatch': nPatientsPerBatch,
            'nBatchesPerEpoch_Training': nBatchesPerEpoch_Training,
            'nBatchesPerEpoch_Validation': nBatchesPerEpoch_Validation,
            'nEpochsPerVideo': nEpochsPerVideo,
            'N': N,
            'loadModelsFlag': loadModelsFlag,
            'enc_chs': enc_chs,
            'dec_chs': dec_chs}

wandb.init(config=config)

model = UNet(enc_chs=enc_chs, dec_chs=dec_chs, pass_ch=2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

alpha1_start_systemA = 1.0
alpha2_start_systemA = 1.0
alpha3_start_systemA = 1.0
alpha4_start_systemA = 0.0
alpha5_start_systemA = 0.0
alpha6_start_systemA = 0.0
alpha7_start_systemA = 0.0
alpha8_start_systemA = 0.0

alpha1_end_systemA = 0.6
alpha2_end_systemA = 0.4
alpha3_end_systemA = 0.2
alpha4_end_systemA = 0.0
alpha5_end_systemA = 0.02
alpha6_end_systemA = 0.05
alpha7_end_systemA = 0.02
alpha8_end_systemA = 0.0

alpha1_start_systemB = 1.0
alpha2_start_systemB = 1.0
alpha3_start_systemB = 1.0
alpha4_start_systemB = 0.0
alpha5_start_systemB = 0.0
alpha6_start_systemB = 0.0
alpha7_start_systemB = 0.0
alpha8_start_systemB = 0.0

alpha1_end_systemB = 0.4
alpha2_end_systemB = 0.4
alpha3_end_systemB = 0.4
alpha4_end_systemB = 0.0
alpha5_end_systemB = 0.03
alpha6_end_systemB = 0.03
alpha7_end_systemB = 0.03
alpha8_end_systemB = 0.0

def exponential_profile(alpha_start, alpha_end, t):
    return alpha_start*((alpha_end/alpha_start)**t)

def linear_profile(alpha_start, alpha_end, t):
    return alpha_start + t*(alpha_end-alpha_start)

def parameters_trajectory_baseline(t):

    t = t*torch.ones([1]).to(device)

    common_weight = torch.zeros([1]).to(device)
    common_weight[:] = 1.0

    alpha1_systemA = exponential_profile(alpha1_start_systemA, alpha1_end_systemA, t)
    alpha2_systemA = exponential_profile(alpha2_start_systemA, alpha2_end_systemA, t)
    alpha3_systemA = exponential_profile(alpha3_start_systemA, alpha3_end_systemA, t)
    alpha4_systemA = linear_profile(alpha4_start_systemA, alpha4_end_systemA, t)
    alpha5_systemA = linear_profile(alpha5_start_systemA, alpha5_end_systemA, t)
    alpha6_systemA = linear_profile(alpha6_start_systemA, alpha6_end_systemA, t)
    alpha7_systemA = linear_profile(alpha7_start_systemA, alpha7_end_systemA, t)
    alpha8_systemA = linear_profile(alpha8_start_systemA, alpha8_end_systemA, t)

    alpha1_common = exponential_profile(0.5*alpha1_start_systemA + 0.5*alpha1_start_systemB, 0.5*alpha1_end_systemA+0.5*alpha1_end_systemB, t)
    alpha2_common = exponential_profile(0.5*alpha2_start_systemA + 0.5*alpha2_start_systemB, 0.5*alpha2_end_systemA+0.5*alpha2_end_systemB, t)
    alpha3_common = exponential_profile(0.5*alpha3_start_systemA + 0.5*alpha3_start_systemB, 0.5*alpha3_end_systemA+0.5*alpha3_end_systemB, t)
    alpha4_common = linear_profile(0.5*alpha4_start_systemA + 0.5*alpha4_start_systemB, 0.5*alpha4_end_systemA+0.5*alpha4_end_systemB, t)
    alpha5_common = linear_profile(0.5*alpha5_start_systemA + 0.5*alpha5_start_systemB, 0.5*alpha5_end_systemA+0.5*alpha5_end_systemB, t)
    alpha6_common = linear_profile(0.5*alpha6_start_systemA + 0.5*alpha6_start_systemB, 0.5*alpha6_end_systemA+0.5*alpha6_end_systemB, t)
    alpha7_common = linear_profile(0.5*alpha7_start_systemA + 0.5*alpha7_start_systemB, 0.5*alpha7_end_systemA+0.5*alpha7_end_systemB, t)
    alpha8_common = linear_profile(0.5*alpha8_start_systemA + 0.5*alpha8_start_systemB, 0.5*alpha8_end_systemA+0.5*alpha8_end_systemB, t)

    alpha1 = alpha1_systemA*(1-common_weight) + alpha1_common*common_weight
    alpha2 = alpha2_systemA*(1-common_weight) + alpha2_common*common_weight
    alpha3 = alpha3_systemA*(1-common_weight) + alpha3_common*common_weight
    alpha4 = alpha4_systemA*(1-common_weight) + alpha4_common*common_weight
    alpha5 = alpha5_systemA*(1-common_weight) + alpha5_common*common_weight
    alpha6 = alpha6_systemA*(1-common_weight) + alpha6_common*common_weight
    alpha7 = alpha7_systemA*(1-common_weight) + alpha7_common*common_weight
    alpha8 = alpha8_systemA*(1-common_weight) + alpha8_common*common_weight

    return alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8

if not loadModelsFlag:
    torch.save(model.state_dict(), 'model_harmonization_baseline.pt')
    torch.save(optimizer.state_dict(), 'optimizer_harmonization_baseline.pt')

for epoch in range(nEpoch):

    data_train = load_training_data_patch(nBatchesPerEpoch_Training*nPatientsPerBatch*nPatchesPerPatient, min=0, max=8000, patch_size=patch_size, patches_per_patient=nPatchesPerPatient)[torch.randperm(nBatchesPerEpoch_Training*nPatientsPerBatch*nPatchesPerPatient)]/3000
    data_validation = load_training_data_patch(nBatchesPerEpoch_Validation*nPatientsPerBatch*nPatchesPerPatient, min=8000, max=10000, patch_size=patch_size, patches_per_patient=nPatchesPerPatient)[torch.randperm(nBatchesPerEpoch_Validation*nPatientsPerBatch*nPatchesPerPatient)]/3000

    loss_training_baseline = 0
    loss_validation_baseline = 0

    loss_ref_training_baseline = 0
    loss_ref_validation_baseline = 0

    # task A training loop
    model.load_state_dict(torch.load('model_harmonization_baseline.pt'))
    optimizer.load_state_dict(torch.load('optimizer_harmonization_baseline.pt'))

    for i in range(nBatchesPerEpoch_Training):

        x = data_train[i*nPatientsPerBatch*nPatchesPerPatient:(i+1)*nPatientsPerBatch*nPatchesPerPatient].to(device)

        x_input = 0*x
        t_channel = 0*x

        for iPatient in range(nPatientsPerBatch*nPatchesPerPatient):
            
            dt = (1/N)*torch.ones([1]).to(device)
            t = np.random.rand()*(1-dt)*torch.ones([1]).to(device) + dt
            t_channel[iPatient,:,:,:] = t

            alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8 = parameters_trajectory_baseline(t)
            
            x_input[iPatient,:,:,:] = apply_NPS(apply_MTF(x[iPatient:(iPatient+1),:,:,:],alpha1,alpha2,alpha3,alpha4), alpha5,alpha6,alpha7,alpha8)

        # concatenate the alphas and the batch to make the input to the network
        unet_input = torch.cat((x_input, t_channel), dim=1)
        
        # forward
        mu_pred = model(unet_input)

        loss = 0
        loss_ref = 0
        for iPatient in range(nPatientsPerBatch*nPatchesPerPatient):

            t = t_channel[iPatient:iPatient+1,0,0,0]
            
            alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8 = parameters_trajectory_baseline(t)
            target_alpha1, target_alpha2, target_alpha3, target_alpha4, target_alpha5, target_alpha6, target_alpha7, target_alpha8 = parameters_trajectory_baseline(t-dt)

            _loss, _mu = kl_divergence_loss(x[iPatient:(iPatient+1)], x_input[iPatient:(iPatient+1)], mu_pred[iPatient:(iPatient+1)], 
                            alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8,
                            target_alpha1, target_alpha2, target_alpha3, target_alpha4, target_alpha5, target_alpha6, target_alpha7, target_alpha8)
            
            loss += (1/(nPatientsPerBatch*nPatchesPerPatient))*_loss

            _loss_ref,_ =  kl_divergence_loss(x[iPatient:(iPatient+1)], x_input[iPatient:(iPatient+1)], x_input[iPatient:(iPatient+1)], 
                            alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8,
                            target_alpha1, target_alpha2, target_alpha3, target_alpha4, target_alpha5, target_alpha6, target_alpha7, target_alpha8)

            loss_ref += (1/(nPatientsPerBatch*nPatchesPerPatient))*_loss_ref

        loss_training_baseline += (1/nBatchesPerEpoch_Training)*loss
        loss_ref_training_baseline += (1/nBatchesPerEpoch_Training)*loss_ref

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: {}, Baseline Training, Iteration: {}, Loss: {}, Loss/Loss_ref: {}'.format(epoch, i, loss.item(), loss.item()/loss_ref.item()))
        
    torch.save(model.state_dict(), 'model_harmonization_baseline.pt')
    torch.save(optimizer.state_dict(), 'optimizer_harmonization_baseline.pt')

    # task A validation loop
    loss_validation_baseline = 0
    for i in range(nBatchesPerEpoch_Validation):

        x = data_validation[i*nPatientsPerBatch*nPatchesPerPatient:(i+1)*nPatientsPerBatch*nPatchesPerPatient].to(device)

        x_input = 0*x
        t_channel = 0*x

        for iPatient in range(nPatientsPerBatch*nPatchesPerPatient):
            
            dt = (1/N)*torch.ones([1]).to(device)
            t = np.random.rand()*(1-dt)*torch.ones([1]).to(device) + dt
            t_channel[iPatient,:,:,:] = t

            alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8 = parameters_trajectory_baseline(t)
            
            x_input[iPatient,:,:,:] = apply_NPS(apply_MTF(x[iPatient:(iPatient+1),:,:,:],alpha1,alpha2,alpha3,alpha4), alpha5,alpha6,alpha7,alpha8)

        
        # concatenate the alphas and the batch to make the input to the network
        unet_input = torch.cat((x_input, t_channel), dim=1)
        
        # forward
        with torch.no_grad():
            mu_pred = model(unet_input)

        loss = 0
        loss_ref = 0
        for iPatient in range(nPatientsPerBatch*nPatchesPerPatient):

            t = t_channel[iPatient:iPatient+1,0,0,0]

            common_weight = torch.zeros([1]).to(device)
            common_weight[t<=0.5] = 1.0
            common_weight[t>0.5] = 2.0 - 2.0*t[t>0.5]

            alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8 = parameters_trajectory_baseline(t)
            target_alpha1, target_alpha2, target_alpha3, target_alpha4, target_alpha5, target_alpha6, target_alpha7, target_alpha8 = parameters_trajectory_baseline(t-dt)

            _loss, _mu = kl_divergence_loss(x[iPatient:(iPatient+1)], x_input[iPatient:(iPatient+1)], mu_pred[iPatient:(iPatient+1)], 
                            alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8,
                            target_alpha1, target_alpha2, target_alpha3, target_alpha4, target_alpha5, target_alpha6, target_alpha7, target_alpha8)
            
            loss += (1/(nPatientsPerBatch*nPatchesPerPatient))*_loss

            _loss_ref,_ =  kl_divergence_loss(x[iPatient:(iPatient+1)], x_input[iPatient:(iPatient+1)], x_input[iPatient:(iPatient+1)], 
                            alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8,
                            target_alpha1, target_alpha2, target_alpha3, target_alpha4, target_alpha5, target_alpha6, target_alpha7, target_alpha8)

            loss_ref += (1/(nPatientsPerBatch*nPatchesPerPatient))*_loss_ref

        loss_validation_baseline += (1/nBatchesPerEpoch_Validation)*loss
        loss_ref_validation_baseline += (1/nBatchesPerEpoch_Validation)*loss_ref

        print('Epoch: {}, Baseline Validation, Iteration: {}, Loss: {}, Loss/Loss_ref: {}'.format(epoch, i, loss.item(), loss.item()/loss_ref.item()))


    # task A video loop
    if epoch % nEpochsPerVideo == 0:
        # run the reverse process
        data_validation = load_training_data(2, min=8000, max=10000)/3000
        x = data_validation[0:1].to(device)
        x[:] = x[0:1]

        x_hat = 0*x
        x_hat_target = 0*x
        t_channel = 0*x

        x_hat_video = np.zeros([N, 1, x.shape[2], x.shape[3]])
        x_hat_target_video = np.zeros([N, 1, x.shape[2], x.shape[3]])

        for n in range(N):

            t = ((N - n)/N)*torch.ones([1]).to(device)

            t_channel[:,:,:,:] = t

            alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8 = parameters_trajectory_baseline(t)
            target_alpha1, target_alpha2, target_alpha3, target_alpha4, target_alpha5, target_alpha6, target_alpha7, target_alpha8 = parameters_trajectory_baseline(t-dt)

            if n == 0:
                for iPatient in range(1):
                    x_hat[iPatient:(iPatient+1),:,:,:] = apply_NPS(apply_MTF(x[0:1], alpha1, alpha2, alpha3, alpha4), alpha5, alpha6, alpha7, alpha8)
                    x_hat_target[iPatient:(iPatient+1),:,:,:] = x_hat[iPatient:(iPatient+1),:,:,:]

            unet_input = torch.cat((x_hat, t_channel), dim=1)

            with torch.no_grad():
                mu_pred = model.eval()(unet_input)

            H_MTF_input = MTF_kernel(alpha1, alpha2, alpha3, alpha4, shape=x.shape)
            H_NPS_input = NPS_kernel(alpha5, alpha6, alpha7, alpha8, shape=x.shape)
            H_MTF_target = MTF_kernel(target_alpha1, target_alpha2, target_alpha3, target_alpha4, shape=x.shape)
            H_NPS_target = NPS_kernel(target_alpha5, target_alpha6, target_alpha7, target_alpha8, shape=x.shape)


            H_LSI, H_epsilon, H_delta, H_NPS_posterior = compute_kl_divergence_parameters(H_MTF_input, H_MTF_target, H_NPS_input, H_NPS_target)

            # have to do this to avoid divide by zero
            # H_NPS_posterior[H_NPS_posterior<1e-6] = 1e-6
            
            x_rfft2 = torch.fft.rfft2(x)
            x_hat_rfft2 = torch.fft.rfft2(x_hat)
            x_hat_target_rfft2 = torch.fft.rfft2(x_hat_target)

            mu_posterior_rfft2 = 0
            # likelihood term
            mu_posterior_rfft2 = mu_posterior_rfft2 + (H_NPS_target/(H_NPS_target + H_delta))*(x_hat_rfft2/H_LSI)
            # prior term
            mu_posterior_rfft2 = mu_posterior_rfft2 + (H_delta/(H_NPS_target + H_delta))* (x_rfft2*H_MTF_target)

            mu_posterior = torch.fft.irfft2(mu_posterior_rfft2)

            mu_posterior_rfft2 = 0
            # likelihood term
            mu_posterior_rfft2 = mu_posterior_rfft2 + (H_NPS_target/(H_NPS_target + H_delta))*(x_hat_target_rfft2/H_LSI)
            # prior term
            mu_posterior_rfft2 = mu_posterior_rfft2 + (H_delta/(H_NPS_target + H_delta))* (x_rfft2*H_MTF_target)
            mu_posterior_target = torch.fft.irfft2(mu_posterior_rfft2)

            z = torch.randn(x.shape).cuda()
            z_rfft2 = torch.fft.rfft2(z)
            z_rfft2 = z_rfft2*torch.sqrt(H_NPS_posterior)
            z = torch.fft.irfft2(z_rfft2)

            x_hat = mu_pred + z
            x_hat_target = mu_posterior_target + z

            x_hat_video[n] = x_hat[:,0].cpu().detach().numpy()
            x_hat_target_video[n] = x_hat_target[:,0].cpu().detach().numpy()

            print('Running Reverse Process n = ', n, ' alpha1 = ', alpha1.item())

        x_hat_target_video = ((x_hat_target_video - 0.2)/(0.5-0.2))
        x_hat_target_video[x_hat_target_video < 0] = 0
        x_hat_target_video[x_hat_target_video > 1] = 1
        x_hat_target_video = 255*x_hat_target_video
        x_hat_target_video = x_hat_target_video[:,0].reshape([N, 1, x.shape[2], x.shape[3]])
        x_hat_target_video_baseline = x_hat_target_video.astype(np.uint8)

        x_hat_video = ((x_hat_video - 0.2)/(0.5-0.2))
        x_hat_video[x_hat_video < 0] = 0
        x_hat_video[x_hat_video > 1] = 1
        x_hat_video = 255*x_hat_video
        x_hat_video = x_hat_video[:,0].reshape([N, 1, x.shape[2], x.shape[3]])
        x_hat_video_baseline = x_hat_video.astype(np.uint8)


        if epoch % nEpochsPerVideo == 0:
            wandb.log({ "epoch": epoch, 
    "Baseline Training Loss": loss_training_baseline.item(), "Baseline Training Loss_ref": loss_ref_training_baseline.item(), "Baseline Training Loss/Loss_ref": loss_training_baseline.item()/loss_ref_training_baseline.item(),
    "Baseline Validation Loss": loss_validation_baseline.item(), "Baseline Validation Loss_ref": loss_ref_validation_baseline.item(), "Baseline Validation Loss/Loss_ref": loss_validation_baseline.item()/loss_ref_validation_baseline.item(),
    "Baseline Reverse Process": wandb.Video(x_hat_video_baseline, fps=16, format="gif"), "Baseline Target Reverse Process": wandb.Video(x_hat_target_video_baseline, fps=16, format="gif")})

    else:
        wandb.log({ "epoch": epoch, 
    "Baseline Training Loss": loss_training_baseline.item(), "Baseline Training Loss_ref": loss_ref_training_baseline.item(), "Baseline Training Loss/Loss_ref": loss_training_baseline.item()/loss_ref_training_baseline.item(),
    "Baseline Validation Loss": loss_validation_baseline.item(), "Baseline Validation Loss_ref": loss_ref_validation_baseline.item(), "Baseline Validation Loss/Loss_ref": loss_validation_baseline.item()/loss_ref_validation_baseline.item()})
