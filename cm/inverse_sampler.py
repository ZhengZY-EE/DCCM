import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import RandomCrop

from .random_util import get_generator
import condition.diffpir_utils.utils_sisr as sr
from abc import ABC, abstractmethod

from cm.karras_diffusion import get_sigmas_karras

__METHOD__ = {}

def register_method(name: str):
    def wrapper(cls):
        if __METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        cls.name = name
        __METHOD__[name] = cls
        return cls
    return wrapper

def get_method(name: str, **kwargs):
    if __METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __METHOD__[name](**kwargs)



class InvrseSampler(ABC):
    def distiller(self, x_t, sigma):
        raise NotImplementedError
        
        

@register_method("cmps")
class InverseSampler_cmps(InvrseSampler):
    def __init__(self, 
                 diffusion,
                 model,
                 operator,
                 measurement,
                 lambda_ = 5.0,
                 eta = 5.0,
                 rho = 7.0,
                 steps=40,
                 clip_denoised=True,
                 sigma_min=0.002,
                 sigma_max=80,
                 acc_grad = False
                 ):
        self.diffusion = diffusion
        self.model = model
        self.operator = operator
        self.measurement = measurement
        self.lambda_ = lambda_
        self.eta = eta
        self.rho = rho
        self.steps = steps
        self.clip_denoised = clip_denoised
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_s = self.operator.sigma_s.clip(min=0.001)
        if self.operator.name not in ["nonlinear_blur","hdr","phase_retrieval"]:
            self.data_solver = __SOLVER__[self.operator.name]
        self.acc_grad = acc_grad
        if hasattr(self.operator, "pre_calculated"):
            self.pre_calculated = operator.pre_calculated
        else :
            self.pre_calculated = None

    def distiller(self, x_t, sigma):
        _, denoised = self.diffusion.denoise(self.model, x_t, sigma)
        if self.clip_denoised:#clip the denoised signal to the range [-1, 1]
            denoised = denoised.clamp(-1, 1)
        return denoised
    

    def loss_grad(self, x, x_t):
        norm = th.linalg.norm(self.measurement - self.operator.forward(x, noiseless=True))
        if self.acc_grad:
            grad = th.autograd.grad(norm, x)[0]
        else:
            grad = th.autograd.grad(norm, x_t)[0]
        return grad


    def distill_one_step(self, sigma, x_t, measurement, tau , rho):
        s_in = x_t.new_ones([x_t.shape[0]])
        x_t = x_t.detach().requires_grad_(True)
        x_0 = self.distiller(x_t, sigma * s_in)
        x_0 = th.clamp(x_0, -1, 1)
        x_0 = x_0.requires_grad_(True)
        x_y_meta = x_0  - self.eta * self.loss_grad(x_0, x_t)
        x_y_0 = self.data_solver(x_y_meta, 
                                 pre_calculated = self.pre_calculated,
                                 operator = self.operator, 
                                 measurement = measurement, 
                                 tau = tau ,
                                 rho = rho,
                                )           

        return x_y_0
    

    def sample_loop(self, shape, device):
        generator = get_generator("dummy")

        #prepare parameters
        sigmas = get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, rho=self.rho, device=device)
        x_var = sigmas.pow(2) / self.lambda_
        self.sigma_s = self.sigma_s.clip(min=0.001)
        rho = self.sigma_s.pow(2) / x_var

        x = generator.randn(*shape, device=device) * self.sigma_max
        s_in = x.new_ones([x.shape[0]])

        index = range(len(sigmas) - 1)

        from tqdm.auto import tqdm
        index = tqdm(index)

        for i in index:
            sigma = sigmas[i]
            rho_i = rho[i]
            tau = rho_i.float().repeat(1, 1, 1, 1)
            x_y_0 = self.distill_one_step(sigma, x, self.measurement, tau, rho_i)
            next_t = sigmas[i+1]

            if i < self.steps - 2:
                x = x_y_0 + generator.randn_like(x_y_0.detach()) * th.sqrt(next_t**2 - self.sigma_min**2)
            
            else:
                x = x_y_0
        
        x = self.distiller(x, self.sigma_min * s_in)

        return x.clamp(-1, 1)
    

@register_method("cmps+opt")
class DCCM_Opt(InverseSampler_cmps):
    def __init__(self,
            diffusion,
            model,
            operator,
            measurement,
            lambda_ = 5.0,
            eta = 5.0,
            rho = 7.0,
            steps=40,
            clip_denoised=True,
            sigma_min=0.002,
            sigma_max=80,
            acc_grad = False,
            op_step = 50 ,
            lr = 5e-3):
        super().__init__(
            diffusion=diffusion,
            model=model,
            operator=operator,
            measurement=measurement,
            lambda_ = lambda_,
            eta = eta,
            rho = rho,
            steps = steps,
            clip_denoised = clip_denoised,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            acc_grad = acc_grad)

        self.op_step = op_step
        self.lr = lr



    def sample_one_step(self, sigma, x_t, measurement,step,rho):
        s_in = x_t.new_ones([x_t.shape[0]])
        x_t = x_t.detach().requires_grad_(True)
        x_0 = self.distiller(x_t, sigma * s_in)
        x_0 = th.clamp(x_0, -1, 1)
        x_0 = x_0.requires_grad_(True)
        x_y_meta = x_0  - self.eta * self.loss_grad(x_0, x_t)
        x = x_y_meta.detach()
        x = th.autograd.Variable(x, requires_grad=True) 

        optimizer =  th.optim.SGD([x], lr=self.lr, weight_decay=0.0)
        
        for i in range(step):
            loss1 = th.linalg.norm(measurement - (self.operator.forward(x, noisless = True)))
            loss2 = th.linalg.norm(x - x_y_meta.detach())
            loss = 1/2 *((loss1**2)  + (rho * loss2**2) )

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        return x
    
    def sample_one_step_approx(self, sigma, x_t, measurement,step,rho):
        s_in = x_t.new_ones([x_t.shape[0]])
        x_t = x_t.detach().requires_grad_(True)
        x_0 = self.distiller(x_t, sigma * s_in)
        x_0 = th.clamp(x_0, -1, 1)
        x_0 = x_0.requires_grad_(True)
        x_y_meta = x_0  - self.eta * self.loss_grad(x_0, x_t)
        norm = th.linalg.norm(self.measurement - self.operator.forward(x_y_meta, noiseless=True))
        norm = norm**2
        x = x_y_meta - 1/2 * (1/rho) * th.autograd.grad(norm, x_y_meta)[0]
        return x
        

    def sample_loop(self, shape, device):
        generator = get_generator("dummy")

        #prepare parameters
        sigmas = get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, rho=self.rho, device=device)
        x_var = sigmas.pow(2) / self.lambda_
        self.sigma_s = self.sigma_s.clip(min=0.001)
        rho = self.sigma_s.pow(2) / x_var


        x = generator.randn(*shape, device=device) * self.sigma_max
        s_in = x.new_ones([x.shape[0]])

        index = range(len(sigmas) - 1)

        from tqdm.auto import tqdm
        index = tqdm(index)

        for i in index:
            sigma = sigmas[i]
            rho_i = rho[i]
            x_y_0 = self.sample_one_step(sigma, x, self.measurement, step=self.op_step,rho=rho_i)
            next_t = sigmas[i+1]

            if i < self.steps - 2:
                x = x_y_0 + generator.randn_like(x_y_0.detach()) * th.sqrt(next_t**2 - self.sigma_min**2)
            
            else:
                x = x_y_0
        
        x = self.distiller(x, self.sigma_min * s_in)

        return x.clamp(-1, 1)



@register_method("cmps+initial")
class InverseSampler_cmpsinit(InvrseSampler):
    def __init__(self, 
                 diffusion,
                 model,
                 operator,
                 measurement,
                 lambda_ = 0.1,
                 eta = 5.0,
                 rho = 7.0,
                 steps=40,
                 clip_denoised=True,
                 sigma_min=0.002,
                 sigma_max=80,
                 acc_grad = False
                 ):
        self.diffusion = diffusion
        self.model = model
        self.operator = operator
        self.measurement = measurement
        self.lambda_ = lambda_
        self.eta = eta
        self.rho = rho
        self.steps = steps
        self.clip_denoised = clip_denoised
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_s = self.operator.sigma_s.clip(min=0.001)
        self.data_solver = __SOLVER__[self.operator.name]
        self.acc_grad = acc_grad
        if hasattr(self.operator, "pre_calculated"):
            self.pre_calculated = operator.pre_calculated
        else :
            self.pre_calculated = None

    def distiller(self, x_t, sigma):
        _, denoised = self.diffusion.denoise(self.model, x_t, sigma)
        if self.clip_denoised:#clip the denoised signal to the range [-1, 1]
            denoised = denoised.clamp(-1, 1)
        return denoised
    

    def loss_grad(self, x, x_t):
        norm = th.linalg.norm(self.measurement - self.operator.forward(x, noiseless=True))
        if self.acc_grad:
            grad = th.autograd.grad(norm, x)[0]
        else:
            grad = th.autograd.grad(norm, x_t)[0]
        return grad


    def distill_one_step(self, sigma, x_t, measurement, tau , rho):
        s_in = x_t.new_ones([x_t.shape[0]])
        x_t = x_t.detach().requires_grad_(True)
        x_0 = self.distiller(x_t, sigma * s_in)
        x_0 = th.clamp(x_0, -1, 1)
        x_0 = x_0.requires_grad_(True)
        x_y_meta = x_0  - self.eta * self.loss_grad(x_0, x_t)
        x_y_0 = self.data_solver(x_y_meta, 
                                 pre_calculated = self.pre_calculated,
                                 operator = self.operator, 
                                 measurement = measurement, 
                                 tau = tau ,
                                 rho = rho,
                                )           

        return x_y_0
    

    def sample_loop(self, shape, device):
        generator = get_generator("dummy")

        #prepare parameters
        sigmas = get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, rho=self.rho, device=device)
        x_var = sigmas.pow(2) / self.lambda_
        self.sigma_s = self.sigma_s.clip(min=0.001)
        rho = self.sigma_s.pow(2) / x_var

        x = self.measurement #+generator.randn(*shape, device=device) * self.sigma_min
        s_in = x.new_ones([x.shape[0]])

        index = range(len(sigmas) - 1)

        from tqdm.auto import tqdm
        index = tqdm(index)

        for i in index:
            sigma = sigmas[i]
            rho_i = rho[i]
            tau = rho_i.float().repeat(1, 1, 1, 1)
            x_y_0 = self.distill_one_step(sigma, x, self.measurement, tau, rho_i)
            next_t = sigmas[i+1]

            if i < self.steps - 2:
                x = x_y_0 + generator.randn_like(x_y_0.detach()) * th.sqrt(next_t**2 - self.sigma_min**2)
            
            else:
                x = x_y_0
        
        x = self.distiller(x, self.sigma_min * s_in)

        return x.clamp(-1, 1)



@register_method("cmps-")
class InverseSampler_cmpsnoacc(InvrseSampler):
    def __init__(self, 
                 diffusion,
                 model,
                 operator,
                 measurement,
                 lambda_ = 5.0,
                 eta = 5.0,
                 rho = 7.0,
                 steps=40,
                 clip_denoised=True,
                 sigma_min=0.002,
                 sigma_max=80,
                 acc_grad = True,
                 step = 5,
                 lr = 1e-4
                 ):
        self.diffusion = diffusion
        self.model = model
        self.operator = operator
        self.measurement = measurement
        self.lambda_ = lambda_
        self.eta = eta
        self.rho = rho
        self.steps = steps
        self.clip_denoised = clip_denoised
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_s = self.operator.sigma_s.clip(min=0.001)
        self.data_solver = __SOLVER__[self.operator.name]
        self.acc_grad = acc_grad
        if hasattr(self.operator, "pre_calculated"):
            self.pre_calculated = operator.pre_calculated
        else :
            self.pre_calculated = None

    def distiller(self, x_t, sigma):
        _, denoised = self.diffusion.denoise(self.model, x_t, sigma)
        if self.clip_denoised:#clip the denoised signal to the range [-1, 1]
            denoised = denoised.clamp(-1, 1)
        return denoised
    




    def distill_one_step(self, sigma, x_t, measurement, tau , rho):
        s_in = x_t.new_ones([x_t.shape[0]])
        x_t = x_t.detach().requires_grad_(True)
        x_0 = self.distiller(x_t, sigma * s_in)
        x_0 = th.clamp(x_0, -1, 1)
        #x_0 = x_0.requires_grad_(True)
        #x_y_meta = x_0  #- self.eta * self.loss_grad(x_0, x_t)
        x_y_0 = self.data_solver(x_0, 
                                 pre_calculated = self.pre_calculated,
                                 operator = self.operator, 
                                 measurement = measurement, 
                                 tau = tau ,
                                 rho = rho,
                                )           

        return x_y_0
    

    def sample_loop(self, shape, device):
        generator = get_generator("dummy")

        #prepare parameters
        sigmas = get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, rho=self.rho, device=device)
        x_var = sigmas.pow(2) / self.lambda_
        self.sigma_s = self.sigma_s.clip(min=0.001)
        rho = self.sigma_s.pow(2) / x_var

        x = generator.randn(*shape, device=device) * self.sigma_max
        s_in = x.new_ones([x.shape[0]])

        index = range(len(sigmas) - 1)

        from tqdm.auto import tqdm
        index = tqdm(index)

        for i in index:
            sigma = sigmas[i]
            rho_i = rho[i]
            tau = rho_i.float().repeat(1, 1, 1, 1)
            x_y_0 = self.distill_one_step(sigma, x, self.measurement, tau, rho_i)
            next_t = sigmas[i+1]

            if i < self.steps - 2:
                x = x_y_0 + generator.randn_like(x_y_0.detach()) * th.sqrt(next_t**2 - self.sigma_min**2)
            
            else:
                x = x_y_0
        
        x = self.distiller(x, self.sigma_min * s_in)

        return x.clamp(-1, 1)


#------------------------------------
# Implementation of proximal solver 
#------------------------------------

__SOLVER__ = {}

def register_solver(name: str):
    def wrapper(func):
        __SOLVER__[name] = func
        return func
    return wrapper


@register_solver("super_resolution")
def super_resolution(x, pre_calculated, operator,measurement,tau, **kwargs):
    FB, FBC, F2B, FBFy = pre_calculated
    sf = operator.scale_factor
    x_y_0 = sr.data_solution(x.float(), FB, FBC, F2B, FBFy, tau, sf)
    return x_y_0


def _debulr(x, pre_calculated, operator, mesurement, tau ,**kwargs):
    FB, FBC, F2B, FBFy = pre_calculated
    x_y_0 = sr.data_solution(x.float(), FB, FBC, F2B, FBFy, tau, 1).float()
    return x_y_0


@register_solver("gaussian_blur")
def gaussian_blur(x, pre_calculated, operator, measurement, tau, **kwargs):
    return _debulr(x, pre_calculated, operator, measurement, tau, **kwargs)


@register_solver("motion_blur")
def motion_blur(x, pre_calculated, operator, measurement, tau, **kwargs):
    return _debulr(x, pre_calculated, operator, measurement, tau, **kwargs)


@register_solver("inpainting")
def inpainting(x, pre_calculated, operator, measurement, tau, rho, **kwargs):
    mask = operator.mask
    return (mask * measurement + x * rho) / (mask + rho)


@register_solver("colorization")
def colorization(x, pre_calculated, operator, measurement, rho, **kwargs):
    d = measurement.repeat(1, 3, 1, 1) / 3 / rho + x
    x_y_0 = d - ((d.mean(dim=[1]).repeat(1, 3, 1, 1) / 3) / (1/3 + rho))

    return x_y_0


