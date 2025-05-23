import torch
import math
import os
import csv
import torch.nn.functional as F
from torch.amp import custom_bwd, custom_fwd
from contextlib import contextmanager
from .utils import default, unnormalize_to_zero_to_one, construct_class_by_name
from einops import rearrange
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from .augment import AugmentPipe
from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.models.autoencoder_retrain import compute_rec_weights
from cldm.loss import compute_layer_weights
import pytorch_lightning as pl
from cldm.loss import SegmentationLoss
from nuScenesSegDataset import nuScenesSegDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tools.training_log_analysis import parse_csv_and_plot

# xt = x0 + ct + \epsilon * t.sqrt()

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.sequential_cross_attn = diff_model_config.pop("sequential_crossattn", False)
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm', 'hybrid-adm', 'crossattn-adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, c_adm=None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            if not self.sequential_cross_attn:
                cc = torch.cat(c_crossattn, 1)
            else:
                cc = c_crossattn
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'hybrid-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm)
        elif self.conditioning_key == 'crossattn-adm':
            assert c_adm is not None
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, y=c_adm)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out
    

class DDPM(pl.LightningModule):
    def __init__(
        self,
        unet_config,
        *,
        image_size,
        conditioning_key=None,
        sampling_timesteps = None,
        loss_type = 'l2',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        clip_x_start=True,
        first_stage_key='image',
        reference_key=None,
        start_dist='normal',
        sample_type='deterministic',
        # perceptual_weight=1.,
        use_l1=False,
        use_ema=True,
        **kwargs
    ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        only_model = kwargs.pop("only_model", False)
        # cfg = kwargs.pop("cfg", None)
        super().__init__()
        # assert not (type(self) == DDPM and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.model =self.model.diffusion_model
        self.channels = self.model.channels
        self.first_stage_key = first_stage_key
        self.reference_key = reference_key
        self.cfg = kwargs
        self.opt_cfg = kwargs.get('trainer_config', None)
        self.data_cfg = kwargs.get('data_config', None)

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = self.opt_cfg is not None
        self.scale_input = self.cfg.get('scale_input', 1)
        self.register_buffer('eps', torch.tensor(self.cfg.get('eps', 1e-4) if self.cfg is not None else 1e-4))
        self.sigma_min = self.cfg.get('sigma_min', 1e-2) if self.cfg is not None else 1e-2
        self.sigma_max = self.cfg.get('sigma_max', 1) if self.cfg is not None else 1
        print('### sigma_min: {}, sigma_max: {} ###\n'.format(self.sigma_min, self.sigma_max))
        self.weighting_loss = self.cfg.get("weighting_loss", False) if self.cfg is not None else False
        if self.weighting_loss:
            print('#### WEIGHTING LOSS ####')

        self.clip_x_start = clip_x_start
        self.image_size = image_size
        self.objective = objective
        self.start_dist = start_dist
        assert start_dist in ['normal', 'uniform']
        # self.sample_type = sample_type

        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, 10)

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # loss_main_cfg_default = {'class_name': 'ddm.loss.MSE_Loss'}
        loss_main_cfg_default = {'class_name': 'ldm.models.autoencoder_retrain.WeightedMSELoss'}
        # loss_vlb_cfg_default = {'class_name': 'ddm.loss.MAE_Loss'}
        loss_main_cfg = self.cfg.get('loss_main', loss_main_cfg_default)
        # loss_vlb_cfg = self.cfg.get('loss_vlb', loss_vlb_cfg_default)
        self.loss_main_func = construct_class_by_name(**loss_main_cfg)
        # self.loss_vlb_func = construct_class_by_name(**loss_vlb_cfg)
        # self.use_l1 = use_l1

        # self.perceptual_weight = perceptual_weight
        # if self.perceptual_weight > 0:
        #     self.perceptual_loss = LPIPS().eval()

        self.use_augment = self.cfg.get('use_augment', False)
        if self.use_augment:
            self.augment = AugmentPipe(p=0.15, xflip=1e8, yflip=1, scale=1, rotate_frac=1,
                                       aniso=1, translate_frac=1)
            print('### use augment ###\n')

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model)
            print(">>> Loaded model from checkpoint:", ckpt_path)

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False, use_ema=False):
        sd = torch.load(path, map_location="cpu")
        if 'ema' in list(sd.keys()) and use_ema:
            sd = sd['ema']
            new_sd = {}
            for k in sd.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]    # remove ema_model.
                    new_sd[new_k] = sd[k]
                else:
                    new_sd[k] = sd[k]
            sd = new_sd
        else:
            if "model" in list(sd.keys()):
                sd = sd["model"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def train_dataloader(self, return_feature=False):
        dataset = nuScenesSegDataset(data_split=self.data_cfg['data_split_train'], 
                                     resolution=self.data_cfg['resolution'],
                                     augment=self.data_cfg['augment'],
                                     return_feature=return_feature)
        
        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=self.trainer.world_size, rank=self.trainer.global_rank, shuffle=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        return DataLoader(dataset, 
                          batch_size=self.data_cfg['batch_size'], 
                          shuffle=shuffle, 
                          sampler=sampler,
                          num_workers=self.data_cfg['num_workers'])
    
    def val_dataloader(self, return_feature=False):
        dataset = nuScenesSegDataset(data_split=self.data_cfg['data_split_val'],
                                     resolution=self.data_cfg['resolution'],
                                     return_feature=return_feature)

        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=self.trainer.world_size, rank=self.trainer.global_rank, shuffle=False)
        else:
            sampler = None
        
        return DataLoader(dataset, 
                          batch_size=self.data_cfg['batch_size'], 
                          sampler=sampler)
    
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, **kwargs):
        z, *_ = self.get_input(self.first_stage_key)
        cond = batch['cond'] if 'cond' in batch else None
        loss, loss_dict = self(z, cond, **kwargs)
        return loss, loss_dict
    
    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def forward(self, x, **kwargs):
        if self.scale_input != 1:
            x = x * self.scale_input
        # continuous time, t in [0, 1]
        eps = self.eps  # smallest time step
        t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
        # t = torch.clamp(t, eps, 1.)
        return self.p_losses(x, t, **kwargs)


    def q_sample(self, x_start, noise, t, C):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x_noisy = x_start + C * time + torch.sqrt(time) * noise
        return x_noisy


    def pred_x0_from_xt(self, xt, noise, C, t):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x0 = xt - C * time - torch.sqrt(time) * noise
        return x0


    def pred_xtms_from_xt(self, xt, noise, C, t, s):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        s = s.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        mean = xt + C * (time-s) - C * time - s / torch.sqrt(time) * noise
        epsilon = torch.randn_like(mean, device=xt.device)
        sigma = torch.sqrt(s * (time-s) / time)
        xtms = mean + sigma * epsilon
        return xtms

    def p_losses(self, x_start, t, *args, **kwargs):
        step = kwargs.get('step', 0)
        ga_ind = kwargs.get("ga_ind", 0)
        if self.start_dist == 'normal':
            noise = torch.randn_like(x_start)
        elif self.start_dist == 'uniform':
            noise = 2 * torch.rand_like(x_start) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        if self.use_augment:
            x_start, aug_label = self.augment(x_start)
            kwargs['augment_labels'] = aug_label
        # K = -1. * torch.ones_like(x_start)
        # C = noise - x_start  # t = 1000 / 1000
        C = -1 * x_start             # U(t) = Ct, U(1) = -x0
        x_noisy = self.q_sample(x_start=x_start, noise=noise, t=t, C=C)  # (b, c, h, w)
        pred = self.model(x_noisy, t, *args, **kwargs)
        C_pred, noise_pred = pred
        # C_pred = C_pred / torch.sqrt(t)
        # noise_pred = noise_pred / torch.sqrt(1 - t)
        # x_rec = self.pred_x0_from_xt(x_noisy, noise_pred, C_pred, t)
        x_rec = -1 * C_pred
        loss_dict = {}
        prefix = 'train'
        target1 = C
        target2 = noise
        target3 = x_start
        loss_simple = 0.
        loss_vlb = 0.
        # use l1 + l2
        if self.weighting_loss:
            simple_weight1 = (t ** 2 - t + 1) / t
            # simple_weight2 = (t ** 2 - t + 1) / (1 - t + self.eps) ** 2 # eps prevents div 0
            simple_weight2 = (t ** 2 - t + 1) / (1 - t + self.eps)  # eps prevents div 0
        else:
            simple_weight1 = 1
            simple_weight2 = 1

        loss_simple += simple_weight1 * self.loss_main_func(C_pred, target1, reduction='sum') + \
                       simple_weight2 * self.loss_main_func(noise_pred, target2, reduction='sum')
        if self.use_l1:
            loss_simple += simple_weight1 * (C_pred - target1).abs().mean([1, 2, 3]) + \
                           simple_weight2 * (noise_pred - target2).abs().mean([1, 2, 3])
            loss_simple = loss_simple / 2
        # rec_weight = 2 * (1 - t.reshape(C.shape[0], 1)) ** 2
        # rec_weight = 1 - t.reshape(C.shape[0], 1) ** 2
        rec_weight = -torch.log(t.reshape(C.shape[0], 1)) / 2
        # loss_simple = loss_simple.sum() / C.shape[0]


        if self.perceptual_weight > 0.:
            loss_vlb += self.perceptual_loss(x_rec, target3).sum([1, 2, 3]) * rec_weight
        # loss_vlb = loss_vlb.sum() / C.shape[0]
        loss = loss_simple.sum() / C.shape[0] + loss_vlb.sum() / C.shape[0]
        loss_dict.update(
            {f'{prefix}/loss_simple': loss_simple.detach().sum() / C.shape[0] / C.shape[1] / C.shape[2] / C.shape[3]})
        loss_dict.update(
            {f'{prefix}/loss_vlb': loss_vlb.detach().sum() / C.shape[0] / C.shape[1] / C.shape[2] / C.shape[3]})
        loss_dict.update({f'{prefix}/loss': loss.detach().sum() / C.shape[0] / C.shape[1] / C.shape[2] / C.shape[3]})
        return loss, loss_dict


    @torch.no_grad()
    def sample(self, batch_size=16, up_scale=1, cond=None, denoise=True):
        image_size, channels = self.image_size, self.channels
        if cond is not None:
            batch_size = cond.shape[0]
        self.sample_type = self.cfg.get('sample_type', 'deterministic')
        if self.sample_type == 'deterministic':
            return self.sample_fn_d((batch_size, channels, image_size[0], image_size[1]),
                                  up_scale=up_scale, unnormalize=True, cond=cond, denoise=denoise)
        elif self.sample_type == 'stochastic':
            return self.sample_fn_s((batch_size, channels, image_size[0], image_size[1]),
                                         up_scale=up_scale, unnormalize=True, cond=cond, denoise=denoise)

    @torch.no_grad()
    def sample_fn_s(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False):
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps
        rho = 1
        step_indices = torch.arange(sampling_timesteps, dtype=torch.float64, device=device)
        t_steps = ((self.sigma_max ** 2) ** (1 / rho) + step_indices / (sampling_timesteps-1) * \
                   ((self.sigma_min ** 2) ** (1 / rho) - (self.sigma_max ** 2) ** (1 / rho))) ** rho
        # t_steps = t_steps * (1 / self.eps).round() * self.eps
        t_steps = torch.cat((t_steps, torch.tensor([0], device=device)), dim=0)
        time_steps = -torch.diff(t_steps)

        if self.start_dist == 'normal':
            img = torch.randn(shape, device=device)
        elif self.start_dist == 'uniform':
            img = 2 * torch.rand(shape, device=device) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        # img = F.interpolate(img, scale_factor=up_scale, mode='bilinear', align_corners=True) * self.sigma_max
        cur_time = torch.ones((batch,), dtype=torch.float64, device=device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((batch,), time_step, device=device)
            if i == time_steps.shape[0] - 1:
                s = cur_time
            if cond is not None:
                pred = self.model(img, cur_time, cond)
            else:
                pred = self.model(img, cur_time)
            # C, noise = pred.chunk(2, dim=1)
            C, noise = pred[:2]
            # correct C
            x0 = self.pred_x0_from_xt(img, noise, C, cur_time)
            if self.clip_x_start:
                x0.clamp_(-1. * self.scale_input, 1. * self.scale_input)
            C = -1 * x0
            img = self.pred_xtms_from_xt(img, noise, C, cur_time, s)
            # img = self.pred_xtms_from_xt2(img, noise, C, cur_time, s)
            cur_time = cur_time - s
        img.clamp_(-1. * self.scale_input, 1. * self.scale_input)
        if self.scale_input != 1:
            img = img / self.scale_input
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample_fn_d(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False):
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps
        rho = 1
        step = 1. / self.sampling_timesteps
        sigma_min = self.sigma_min ** 2
        # sigma_min = step
        step_indices = torch.arange(sampling_timesteps, dtype=torch.float64, device=device)
        # t_steps = ((self.sigma_max ** 2) ** (1 / rho) + step_indices / (sampling_timesteps) * \
        #            ((self.sigma_min ** 2) ** (1 / rho) - (self.sigma_max ** 2) ** (1 / rho))) ** rho
        t_steps = (self.sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
                sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        # time_steps = -torch.diff(t_steps)
        alpha = 1

        x_next = torch.randn(shape, device=device, dtype=torch.float64) * t_steps[0]
        # cur_time = torch.ones((batch,), device=device)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # t_cur = torch.full((batch,), t_c, device=device)
            # t_next = torch.full((batch,), t_n, device=device)  # 0, ..., N-1
            x_cur = x_next
            if cond is not None:
                pred = self.model(x_cur, t_cur, cond)
            else:
                pred = self.model(x_cur, t_cur)
            C, noise = pred[:2]
            C, noise = C.to(torch.float64), noise.to(torch.float64)
            x0 = x_cur - C * t_cur - noise * t_cur.sqrt()
            if self.clip_x_start:
                x0.clamp_(-1. * self.scale_input, 1. * self.scale_input)
            x_next = x0 + C * t_next + noise * t_next.sqrt()
            # x_next = x0 + t_next * C + t_next.sqrt() * noise
            # x0 = self.pred_x0_from_xt(x_cur, noise, C, t_cur)
            # d_cur = C + noise / t_cur.sqrt()

            # Apply 2-order correction.
            # if i < sampling_timesteps - 1:
            #     if cond is not None:
            #         pred = self.model(x_next, t_next, cond)
            #     else:
            #         pred = self.model(x_next, t_next)
            #     C_, noise_ = pred[:2]
            #     C_, noise_ = C_.to(torch.float64), noise_.to(torch.float64)
            #     d_next = C_ + noise_ / (t_next.sqrt() + t_next.sqrt())
            #     x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_next)

        x_next.clamp_(-1. * self.scale_input, 1. * self.scale_input)
        if self.scale_input != 1:
            x_next = x_next / self.scale_input
        if unnormalize:
            x_next = unnormalize_to_zero_to_one(x_next)
        return x_next
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")


class LatentDiffusion(DDPM):
    def __init__(self,
                 first_stage_config,
                 scale_factor=1.0,
                 scale_by_std=True,
                 scale_by_softsign=False,
                 sample_type='deterministic',
                 default_scale=False,
                 *args,
                 **kwargs
                 ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        only_model = kwargs.pop("only_model", False)
        super().__init__(*args, **kwargs)
        self.scale_factor = scale_factor
        self.clip_denoised = False
        # self.sample_type = sample_type

        if self.cfg.get('use_disloss', False):
            loss_dis_func_default = {'class_name': 'ddm.loss.MAE_Loss'}
            loss_dis_func = self.cfg.get('loss_dis', loss_dis_func_default)
            self.loss_dis_func = construct_class_by_name(**loss_dis_func)
        self.loss_seg_func = SegmentationLoss()

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model)
            self.restarted_from_ckpt = True
        else:
            self.apply(self.init_weights)
            self.instantiate_first_stage(first_stage_config)


    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train

        ckpt_path = config.get('ckpt_path', None)
        if ckpt_path:
            vae_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.first_stage_model.load_state_dict(vae_ckpt["state_dict"])

        # Disable gradient update of the VAE during LDM training
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
    


    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)  # Use Xavier initialization
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z.detach()
    

    def configure_optimizers(self):

        base_lr = self.opt_cfg.get("lr", 5e-5)
        min_lr = self.opt_cfg.get("min_lr", 1e-7)
        weight_decay = self.opt_cfg.get("weight_decay", 1e-4)

        train_loader = self.train_dataloader()
        steps_per_epoch = len(train_loader)
        num_epochs = self.trainer.max_epochs
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(0.05 * total_steps)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.get_optim_params()),
            lr=base_lr,
            weight_decay=weight_decay
        )

        # scheduler (linear warmup + cosine decay)
        def lr_lambda(current_step):
            if current_step <= warmup_steps:
                return float(current_step + 1) / float(warmup_steps)
            else:
                decay_step = current_step - warmup_steps
                decay_total = max(total_steps - warmup_steps, 1)
                decay_ratio = (1.0 - decay_step / decay_total) ** 0.96
                min_ratio = min_lr / base_lr
                return max(decay_ratio, min_ratio)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        print(f"Training steps: {total_steps}, Warmup: {warmup_steps}, Steps per epoch: {steps_per_epoch}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_cosine"
            }
        }
    

    def get_optim_params(self):
        return self.parameters()


    @torch.no_grad()
    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics  # All logged metrics
        epoch = int(self.current_epoch)

        fields = [
            "train/loss_epoch",
            "train/loss_simple_epoch",
            "train/loss_vlb_epoch",
            "train/loss_seg_epoch",
            "train/loss_c_epoch",
            "train/loss_o_epoch",
            "val/loss",
            "val/loss_simple",
            "val/loss_vlb",
            "val/loss_seg",
            "val/loss_c",
            "val/loss_o"
        ]

        row = {"epoch": epoch}
        for key in fields:
            val = metrics.get(key)
            row[key] = val.item() if val is not None else None

        # Write to CSV
        csv_file = os.path.join(self.logger.log_dir, "train_log.csv")
        file_exists = os.path.exists(csv_file)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["epoch"] + fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: ("" if v is None else f"{v:.3f}") for k, v in row.items()})

        # Draw learning curve
        learning_curve = os.path.join(self.logger.log_dir, "loss_plot.png")
        parse_csv_and_plot(csv_file, learning_curve)


    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, bs=None):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        
        if self.reference_key is not None:
            ref = super().get_input(batch, self.reference_key)
            if bs is not None:
                ref = ref[:bs]
            ref = ref.to(self.device)
            encoder_posterior = self.first_stage_model.encode(ref)
            ref = self.get_first_stage_encoding(encoder_posterior)
        else:
            ref = x

        encoder_posterior = self.first_stage_model.encode(x)
        z = self.get_first_stage_encoding(encoder_posterior)

        cond = None
        out = [z, cond, ref]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        return out

    def training_step(self, batch, **kwargs):

        loss, loss_dict = self.shared_step(batch, prefix='train', **kwargs)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)

        if torch.isnan(loss):
            print(f" !!!!!!!!!! NaN detected in loss at step {self.global_step} !!!!!!!!!!")
            print(f"Step: {self.global_step}, LR: {lr}, Loss Dict: {loss_dict}")
            raise ValueError("Training diverged due to NaN")

        return loss
    
    def validation_step(self, batch, **kwargs):
        val_epoch = self.opt_cfg.get('val_after_epoch', 0)

        if self.current_epoch < val_epoch:
            self.log("val/loss", torch.tensor(float("nan"), device=self.device), 
                     prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
            return
        
        loss, loss_dict = self.shared_step(batch, prefix='val', **kwargs)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)

    def predict_step(self, batch):
        pass

    
    def apply_model(self, x_noisy, t, cond, **kwargs):
        cond = {'c_concat': cond}
        C_pred, noise_pred = self.model(x_noisy, t, **cond)
        return C_pred, noise_pred


    def p_losses(self, x_start, t, cond, *args, **kwargs):
        if self.start_dist == 'normal':
            noise = torch.randn_like(x_start)
        elif self.start_dist == 'uniform':
            noise = 2 * torch.rand_like(x_start) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        # K = -1. * torch.ones_like(x_start)
        # C = noise - x_start  # t = 1000 / 1000
        C = -1 * x_start             # U(t) = Ct, U(1) = -x0
        # C = -2 * x_start               # U(t) = 1/2 * C * t**2, U(1) = 1/2 * C = -x0
        x_noisy = self.q_sample(x_start=x_start, noise=noise, t=t, C=C)  # (b, 2, c, h, w)
        C_pred, noise_pred = self.apply_model(x_noisy, t, cond, **kwargs)
        x_rec = self.pred_x0_from_xt(x_noisy, noise_pred, C_pred, t)
        loss_dict = {}
        prefix = kwargs.get('prefix', 'train')

        target1 = C
        target2 = noise
        target3 = x_start
        loss = 0.
        # loss_vlb = 0.
        # use l1 + l2
        # if self.weighting_loss:
        #     simple_weight1 = (t ** 2 - t + 1) / t
        #     # simple_weight2 = (t ** 2 - t + 1) / (1 - t + self.eps) ** 2 # eps prevents div 0
        #     simple_weight2 = (t ** 2 - t + 1) / (1 - t + self.eps)  # eps prevents div 0
        # else:
        #     simple_weight1 = 1
        #     simple_weight2 = 1
        # loss_simple += simple_weight1 * self.loss_main_func(C_pred, target1, reduction='sum') + \
        #                simple_weight2 * self.loss_main_func(noise_pred, target2, reduction='sum')
        
        ### calculate the weights for each layer
        # img_ori = kwargs['batch'][self.first_stage_key].permute(0, 3, 1, 2)
        # bs, num_layers, h, w = img_ori.shape
        # total_pixel = h * w
        # layer_pixel = ((img_ori > 0.99) & (img_ori < 1.01)).sum(dim=(2, 3))    # calculate the sum of pixel values of each layer
        # weights = torch.log(total_pixel / (layer_pixel + 1e-3))  # [bs, 4]
        # weights = weights ** 2
        # normalized_weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        # rec_weight = normalized_weights.view(bs, num_layers, 1, 1)     # # [bs, 4]->[bs, 4, 1, 1]
        # rec_weight = compute_rec_weights(img_ori)
        
        ## l2 loss
        # loss_simple = self.loss_main_func(C_pred, target1, rec_weight) + \
        #                 self.loss_main_func(noise_pred, target2, rec_weight)
        loss_simple = F.mse_loss(C_pred, target1, reduction='mean') + \
                        F.mse_loss(noise_pred, target2, reduction='mean')
        loss += loss_simple
        
        ## l1 loss
        loss_vlb = (x_rec - target3).abs().mean()
        loss += loss_vlb

        ## Segmentation loss
        img_ori = kwargs['batch'][self.first_stage_key].permute(0, 3, 1, 2)
        # rec_weight = compute_rec_weights(img_ori)
        rec_weight = compute_layer_weights(img_ori)
        img_rec = self.decode_first_stage(x_rec)
        loss_seg = self.loss_seg_func(img_rec, img_ori, rec_weight) * 0.3
        loss += loss_seg

        loss_dict.update({f'{prefix}/loss_simple': loss_simple.detach()})
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb.detach()})
        loss_dict.update({f'{prefix}/loss_seg': loss_seg.detach()})
        loss_dict.update({f'{prefix}/loss': loss.detach()})

        return loss, loss_dict

    
    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)
    
    def shared_step(self, batch, **kwargs):
        kwargs['batch'] = batch
        x, cond, ref = self.get_input(batch, self.first_stage_key)
        loss, loss_dic = self(x, cond=cond, ref=ref, **kwargs)
        return loss, loss_dic

    @torch.no_grad()
    def sample(self, batch_size=16, cond=None, denoise=True):
        image_size, channels = self.image_size, self.channels
        # if cond is not None:
        #     batch_size = cond.shape[0]
        down_ratio = 2 ** (len(self.first_stage_model.ddconfig['ch_mult']) - 1)
        # down_ratio = self.first_stage_model.down_ratio
        self.sample_type = self.cfg.get('sample_type', 'deterministic')
        if self.sample_type == 'deterministic':
            z = self.sample_fn_d((batch_size, channels, image_size[0]//down_ratio, 
                                  image_size[1]//down_ratio), cond=cond, denoise=denoise)
        elif self.sample_type == 'stochastic':
            z = self.sample_fn_s((batch_size, channels, image_size[0]//down_ratio, 
                                  image_size[1]//down_ratio), cond=cond, denoise=denoise)

        # if self.scale_by_std:
        #     z = 1. / self.scale_factor * z.detach()
        # elif self.scale_by_softsign:
        #     z = z / (1 - z.abs())
        #     z = z.detach()

        x_rec = self.decode_first_stage(z.to(torch.float32))

        return x_rec

    @torch.no_grad()
    def sample_fn_s(self, shape, cond=None, denoise=False):
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps
        rho = 1
        step = 1. / self.sampling_timesteps
        sigma_min = self.sigma_min ** 2
        step_indices = torch.arange(sampling_timesteps, dtype=torch.float64, device=device)
        # t_steps = ((self.sigma_max ** 2) ** (1 / rho) + step_indices / (sampling_timesteps) * \
        #            ((self.sigma_min ** 2) ** (1 / rho) - (self.sigma_max ** 2) ** (1 / rho))) ** rho
        t_steps = (self.sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
                sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        time_steps = -torch.diff(t_steps)

        if self.start_dist == 'normal':
            img = torch.randn(shape, device=device)
        elif self.start_dist == 'uniform':
            img = 2 * torch.rand(shape, device=device) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        # K = -1 * torch.ones_like(img)
        cur_time = torch.ones((batch,), device=device)
        model_fn = lambda img, t: self.apply_model(img, t, cond) if cond is not None else self.apply_model(img, t)
        for i, time_step in enumerate(time_steps):
            s = torch.full((batch,), time_step, device=device)
            if i == time_steps.shape[0] - 1:
                s = cur_time
            C, noise = model_fn(img, cur_time)

            # correct C
            x0 = self.pred_x0_from_xt(img, noise, C, cur_time)
            C = -1 * x0
            img = self.pred_xtms_from_xt(img, noise, C, cur_time, s)
            # img = self.pred_xtms_from_xt2(img, noise, C, cur_time, s)
            cur_time = cur_time - s

        return img

    @torch.no_grad()
    def sample_fn_d(self, shape, cond=None, denoise=False):
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps

        sigma_min = self.sigma_min ** 2
        rho = 1.
        step_indices = torch.arange(sampling_timesteps, dtype=torch.float64, device=device)
        t_steps = (self.sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
                sigma_min - self.sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        x_next = torch.randn(shape, device=device, dtype=torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            C, noise = self.apply_model(x_cur, t_cur, cond)
            C, noise = C.to(torch.float64), noise.to(torch.float64)
            # x0 = x_cur - C * t_cur - noise * t_cur.sqrt()
            d_cur = C + noise / (t_cur.sqrt() + t_next.sqrt())
            x_next = x_cur + (t_next - t_cur) * d_cur

        img = x_next

        return img
    

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True,**kwargs):

        log = dict()
        z, c, ref = self.get_input(batch, self.first_stage_key, bs=N)
        # c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)

        log["recon_sd_input"] = self.decode_first_stage(z)    

        if sample:
            samples = self.sample(batch_size=N)
            # x_samples = self.decode_first_stage(samples)
            log["samples"] = samples
            # log["samples"] = self.render_stp3_prediction(x_samples) # decode with stp3 decoder
        return log

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones(input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None