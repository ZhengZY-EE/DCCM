"""
Solve inverse problems with consistency model
"""

import argparse
import os

import numpy as np
import torch as th
from torchvision import transforms

import sys
sys.path.append('')

from cm import dist_util, logger, utils
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.inverse_sampler import get_method

from condition.measurements import get_operator

import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pytorch_fid import fid_score
import lpips
import tqdm

from functools import partial

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_yaml(data: dict, file_path: str):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)

def compute_metrics(pred_x0, x0, loss_fn_vgg):
    def to_eval(x: th.Tensor):
        return (x[0] / 2 + 0.5).clip(0, 1).detach()
    psnr = peak_signal_noise_ratio(to_eval(x0).cpu().numpy(), to_eval(pred_x0).cpu().numpy(), data_range=1).item() #must give the datarange
    ssim = structural_similarity(to_eval(x0).cpu().numpy(), to_eval(pred_x0).cpu().numpy(), channel_axis=0, data_range=1).item()
    lpips = loss_fn_vgg((x0).clip(-1,1),(pred_x0).clip(-1,1))[0, 0, 0, 0].item()
    metrics = {'psnr': psnr, 'ssim': ssim, 'lpips': lpips}
    #print(metrics)
    return metrics

def compute_fid(sample_root, pred_root, device):
    paths = [str(sample_root),str(pred_root)]
    fid = fid_score.calculate_fid_given_paths(paths, batch_size = 100, device=device,dims=2048,num_workers=0)
    return fid

def calculate_average_metric(metrics_list):
    avg_dict = {}
    count_dict = {}

    for metrics in metrics_list:
        for key, value in metrics.items():
            if key not in avg_dict:
                avg_dict[key] = 0.0
                count_dict[key] = 0
            avg_dict[key] += value
            count_dict[key] += 1

    for key in avg_dict:
        if count_dict[key] > 0:
            avg_dict[key] /= count_dict[key]

    return avg_dict


def main():
#-----------------------------------------
# Define arguments
#-----------------------------------------
    parser = create_argparser()                            
    parser.add_argument('--batch_size', type=int, default=1, help='the batch_size')
    parser.add_argument('--ckpt', type=str, default="../model/lsun_bed_256.pt")
    parser.add_argument('--task_config', type=str, default="configs/inpainting_config.yaml")
    parser.add_argument('--datasets_path', type=str, default="data/samples")
    parser.add_argument('--save_img',dest='save_img', action='store_true')
    parser.add_argument('--logdir', type=str, default='results/lsun_bed_256/super_res')
    parser.add_argument('--method', type=str, default='cmps')

#-----------------------------------------
# Setup consistency model and test data
#-----------------------------------------
    opt = parser.parse_args()
        # set random seed
    
    device = dist_util.dev()
    
    th.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(opt.seed)

    assert opt.batch_size and opt.batch_size == 1, "only batch_size=1 is supported for now"

    dist_util.setup_dist()
    #logger.configure()

    if "consistency" in opt.training_mode:
        distillation = True
    else:
        distillation = False

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(opt, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(opt.ckpt, map_location="cpu")
    )
    model.to(dist_util.dev())
    if opt.use_fp16:
        model.convert_to_fp16()
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(), lambda x: x*2-1])  #lambda x: x*2-1
    test_data = utils.FolderOfImages(opt.datasets_path, transform=transform)
    test_dataloader = th.utils.data.DataLoader(test_data, batch_size=opt.batch_size)
    
#------------------------------------
# Image Process and Inference Stage
#------------------------------------
    device = dist_util.dev()

    task_config = load_yaml(opt.task_config)
    operator = get_operator(device = device, **task_config)
    #print("kernel:", operator.kernel)
    #print(operator.kernel.shape)
    
    #print(f"Operation: {task_config['name']} / sigma_s: {task_config['sigma_s']}")

    #print("sampling...")

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    def sample(step):

        if not os.path.exists(opt.logdir):
            os.makedirs(opt.logdir) 

        #loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        #sigmas = get_sigmas_karras(args.steps, args.sigma_min, args.sigma_max, rho=7., device=device)
        metrics_list = []

        for i, img in enumerate((test_dataloader)):
            print(f"sampling {i + 1}th image...")
            x0, = img
            x0 = x0.to(device)
            measurements = operator.forward(x0.clone()) #ensure x0 is not changed

            args = dict(diffusion = diffusion,
                        model = model,
                        operator = operator,
                        measurement = measurements,
                        lambda_ = 2 ,#3.0
                        eta = 5, #1.0
                        rho = 7.0, 
                        steps = step, #step = opt.steps
                        clip_denoised = True,
                        sigma_min = 0.002,      
                        sigma_max = 80,
                        acc_grad = True,
                    
                        )
            
            method = get_method(opt.method, **args)
            sample_fn = partial(method.sample_loop, shape=x0.shape, device=device)
            
            x0_pred = sample_fn()

            #quantitative metrics
            metrics = compute_metrics(x0_pred, x0, loss_fn_vgg)
            metrics_list.append(metrics)
            
            #save images
            if opt.save_img:
                measurements_path = os.path.join(opt.logdir,f"step:{step}", "images", "measurements")
                pred_path = os.path.join(opt.logdir,f"step:{step}", "images", "pred")
                if not os.path.exists(measurements_path):
                    os.makedirs(measurements_path)
                if not os.path.exists(pred_path):
                    os.makedirs(pred_path)
                base_count = len(os.listdir(measurements_path))
                measurements_filename = os.path.join(measurements_path, f"y_{base_count+1}.png")
                utils.to_pil_image(measurements).save(measurements_filename)
                x0_pred_filename = os.path.join(pred_path , f"pred_{base_count+1}.png")
                utils.to_pil_image(x0_pred).save(x0_pred_filename)
            #qualitive results

        avg_metrics = calculate_average_metric(metrics_list)
        print(avg_metrics)

        from pathlib import Path
        label_root = Path(f'./data/lsun_cat_val100') #lsun_bed_val100
        recon_root = pred_path  #gaussian_deblur, super_resolution, inpainting
        paths = [str(label_root),str(recon_root)]

        fid = fid_score.calculate_fid_given_paths(paths, batch_size =1, device=device, dims=2048, num_workers=0)
        avg_metrics['fid'] = fid
        #print(f'FID: {fid}')
        print(avg_metrics)

        metric_path = os.path.join(opt.logdir, "metrics")
        if not os.path.exists(metric_path):
            os.makedirs(metric_path)
        metric_count = len(os.listdir(metric_path))
        save_yaml(avg_metrics, os.path.join(metric_path, f"avg_metrics_{metric_count}.yaml"))
    
    try:
        #for eta in [0]:
            #for lambda_ in [3]:
            #for lambda_ in range(1,10):
        for step in [40]:
            print("step:", step)
            #sample(eta, lambda_)
            sample(step)
        #logger.log("sampling complete")
        print("sampling complete")
    except KeyboardInterrupt:
        pass


# args_utilizes
def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        #batch_size=1,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=39,
        ts="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
