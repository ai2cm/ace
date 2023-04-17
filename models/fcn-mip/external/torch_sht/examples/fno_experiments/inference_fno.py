import os
import argparse as ap
import time
from tqdm import tqdm

import numpy as np
import torch

import matplotlib.pyplot as plt

from train_fno import l2loss_sphere 

# rolls out the FNO and compares to the classical solver
def autoregressive_inference(model, dataset, path_root, nsteps, autoreg_steps=10, nskip=1, plot_channel=0, nics=20):

    model.eval()

    losses = np.zeros(nics)
    fno_times = np.zeros(nics)
    nwp_times = np.zeros(nics)
    
    with torch.no_grad():
        for iic in tqdm(range(nics)):
            ic = dataset.solver.random_initial_condition(mach=0.2)
            inp_mean = dataset.inp_mean
            inp_var = dataset.inp_var

            prd = (dataset.solver.spec2grid(ic) - inp_mean) / torch.sqrt(inp_var)
            prd = prd.unsqueeze(0)
            uspec = ic.clone()

            # ML model
            start_time = time.time()
            for i in range(autoreg_steps+1):
                # evaluate the ML model
                prd = model(prd)

                if (nskip > 0) and (i % nskip == 0):

                    # do plotting
                    fig = plt.figure(figsize=(7.5, 6))
                    dataset.solver.plot_griddata(prd[0, plot_channel], fig, vmax=4, vmin=-4)
                    plt.savefig(os.path.join(path_root, 'pred_'+str(i//nskip)+'.png'))
                    plt.close(fig)
                    plt.cla()
                    plt.clf()

            fno_times[iic] = time.time() - start_time

            # classical model
            start_time = time.time()
            for i in range(autoreg_steps+1):
            
                # advance classical model
                uspec = dataset.solver.timestep(uspec, nsteps)
                
                if (nskip > 0) and (i % nskip == 0):
                    ref = (dataset.solver.spec2grid(uspec) - inp_mean) / torch.sqrt(inp_var)
                    fig = plt.figure(figsize=(7.5, 6))
                    dataset.solver.plot_griddata(ref[plot_channel], fig, vmax=4, vmin=-4)
                    plt.savefig(os.path.join(path_root, 'truth_'+str(i//nskip)+'.png'))
                    plt.close(fig)
                    plt.cla()
                    plt.clf()

            nwp_times[iic] = time.time() - start_time
            ref = (dataset.solver.spec2grid(uspec) - inp_mean) / torch.sqrt(inp_var)
            losses[iic] = torch.sqrt(l2loss_sphere(dataset.solver, prd, ref, True).sum()).item()
        

    return losses, fno_times, nwp_times

def scrub_state_dict(state_dict):
    scrubbed_state_dict = {}

    exclude_list = [".weights", ".pct", "pos_embed"]
    for k,v in state_dict.items():
        if not any([x in k for x in exclude_list]):
            scrubbed_state_dict[k] = v

    return scrubbed_state_dict
    
def main(args):
    
    # cfreate output dir if it does not exist:
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.isfile(args.checkpoint_file):
        raise FileNotFoundError(f"Error, checkpoint {args.checkpoint_file} not found.")

    # set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333) 
    
    # set device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)
    
    # create model:
    from utils.fno_layers import FourierNeuralOperatorNet
    model = FourierNeuralOperatorNet(spectral_transform=args.spectral_transform, 
                                     filter_type=args.filter_type, img_size=(args.nlat, args.nlon),
                                     num_layers=args.num_layers, scale_factor=args.scale_factor, 
                                     embed_dim=args.embed_dim, complex_activation=args.complex_activation,
                                     normalization_layer=args.normalization_layer,
                                     pos_embedding=args.enable_pos_embedding)

    print(model)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint_file, map_location="cpu")
    checkpoint = scrub_state_dict(checkpoint)
    model.load_state_dict(checkpoint)
    model.to(device)
    
    # dataloader
    from utils.pde_dataset import PdeDataset
    dt = 1*3600
    dt_solver = 150
    nsteps = dt // dt_solver
    dataset = PdeDataset(dt=dt, nsteps=nsteps, dims=(args.nlat, args.nlon), device=device, normalize=True)
    
    # inference
    losses, fno_times, nwp_times = autoregressive_inference(model, dataset, args.output_dir, nsteps=nsteps, autoreg_steps=args.autoreg_steps)
    
    print(f"Average loss: {losses.mean()}")
        

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('forkserver', force=True)
    
    # set up parser
    parser = ap.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Directory which outputs are going to be written to")
    parser.add_argument("--checkpoint_file", type=str, default=None, help="Full path to checkpoint file to perform inference on.")
    # analysis parameters
    parser.add_argument("--autoreg_steps", type=int, default=10, help ="Number of autoregression steps.")
    # model parameters
    parser.add_argument("--nlat", type=int, default=256, help ="Number of latitude points.")
    parser.add_argument("--nlon", type=int, default=512, help ="Number of longitude points.")
    parser.add_argument("--num_layers", type=int, required=True, help ="Number of layers.")
    parser.add_argument("--scale_factor", type=int, required=True, help ="Scale factor for spectral projection.")
    parser.add_argument("--embed_dim", type=int, required=True, help ="Embedding dim size.")
    parser.add_argument("--spectral_transform", type=str, default="sht", choices=["sht", "fft"], help="Which type of transform do we want.")
    parser.add_argument("--filter_type", type=str, default="non-linear", choices=["non-linear", "linear"], help="Which type of spectral filter do we want.")
    parser.add_argument("--normalization_layer", type=str, default="instance_norm", choices=["instance_norm", "layer_norm"], help="Specify type of normalization layer.") 
    parser.add_argument("--complex_activation", type=str, default="real", choices=["real"], help="Which type of complex activation do we want.")
    parser.add_argument("--enable_pos_embedding", action='store_true')
    # parse
    args = parser.parse_args()

    # do eval
    main(args)
