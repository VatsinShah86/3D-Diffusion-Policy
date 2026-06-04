from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder

class DP3(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            one_hot_seg=False,
            seg_channel_idx=6,
            num_seg_classes=3,
            separate_gripper_head=False,
            lambda_gripper=1.0,
            gripper_pos_weight=1.9,
            gripper_condition_on_prev=True,
            gripper_prev_dropout=0.5,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        self.separate_gripper_head = separate_gripper_head
        self.gripper_action_idx = 6
        self.gripper_width_obs_idx = 6
        self.lambda_gripper = lambda_gripper
        self.gripper_threshold = 0.2
        self.gripper_condition_on_prev = gripper_condition_on_prev
        self.gripper_prev_dropout = gripper_prev_dropout
        self.register_buffer('gripper_pos_weight', torch.tensor(float(gripper_pos_weight)))
        self.diffusion_action_dim = (action_dim - 1) if separate_gripper_head else action_dim

        if separate_gripper_head:
            assert obs_as_global_cond, "separate_gripper_head requires obs_as_global_cond=True"

        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        obs_encoder = DP3Encoder(observation_space=obs_dict,
                                                   img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                one_hot_seg=one_hot_seg,
                                                seg_channel_idx=seg_channel_idx,
                                                num_seg_classes=num_seg_classes,
                                                )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = self.diffusion_action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = self.diffusion_action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")



        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler

        self.gripper_head = None
        if self.separate_gripper_head:
            if obs_as_global_cond and "cross_attention" not in self.condition_type:
                gripper_in_dim = obs_feature_dim * n_obs_steps
            else:
                gripper_in_dim = obs_feature_dim
            self.gripper_extra_dim = 2 if self.gripper_condition_on_prev else 0
            gripper_in_dim = gripper_in_dim + self.gripper_extra_dim
            self.gripper_head = nn.Sequential(
                nn.Linear(gripper_in_dim, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, n_action_steps),
            )

        # Inference-time running state: gripper command committed last chunk.
        self._prev_gripper_cmd = None

        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.diffusion_action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


        print_params(self)
        
    def reset(self):
        super().reset()
        self._prev_gripper_cmd = None

    def _gripper_head_input(self, global_cond, prev_gripper, curr_width):
        """Concatenate global_cond with prev_gripper and curr_width scalars."""
        if not self.gripper_condition_on_prev:
            return global_cond
        prev = prev_gripper.reshape(-1, 1).to(global_cond.dtype)
        width = curr_width.reshape(-1, 1).to(global_cond.dtype)
        return torch.cat([global_cond, prev, width], dim=-1)

    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]


            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.diffusion_action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        if self.separate_gripper_head:
            pose_pred = self._unnormalize_pose(nsample[..., :Da])  # (B, T, 6)
            action_pred = torch.zeros(B, T, self.action_dim,
                                      device=pose_pred.device, dtype=pose_pred.dtype)
            action_pred[..., :self.gripper_action_idx] = pose_pred
        else:
            naction_pred = nsample[..., :Da]
            action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end].clone()
        if self.separate_gripper_head:
            curr_width = nobs['agent_pos'][:, self.n_obs_steps - 1, self.gripper_width_obs_idx]
            if self.gripper_condition_on_prev:
                if self._prev_gripper_cmd is None:
                    self._prev_gripper_cmd = torch.ones(B, device=global_cond.device, dtype=global_cond.dtype)
                head_in = self._gripper_head_input(global_cond, self._prev_gripper_cmd, curr_width)
            else:
                head_in = global_cond
            gripper_logits = self.gripper_head(head_in)              # (B, n_action_steps)
            gripper_bin = (torch.sigmoid(gripper_logits) > 0.5).float()
            action[..., self.gripper_action_idx] = gripper_bin
            if self.gripper_condition_on_prev:
                self._prev_gripper_cmd = gripper_bin[:, -1].detach()

        result = {
            'action': action,
            'action_pred': action_pred,
        }

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _unnormalize_pose(self, naction_pose: torch.Tensor) -> torch.Tensor:
        """Unnormalize the 6 pose dims by padding a dummy gripper column, then slicing."""
        B, T, _ = naction_pose.shape
        pad = torch.zeros(B, T, 1, device=naction_pose.device, dtype=naction_pose.dtype)
        full = torch.cat([naction_pose, pad], dim=-1)   # (B, T, 7)
        return self.normalizer['action'].unnormalize(full)[..., :self.gripper_action_idx]

    def compute_loss(self, batch):
        # normalize input

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.separate_gripper_head:
            gidx = self.gripper_action_idx
            trajectory = nactions[..., :gidx]                          # (B, T, 6) pose only
            raw_gripper = batch['action'][..., gidx]                   # (B, T) raw command
            gripper_target = (raw_gripper > self.gripper_threshold).float()
        else:
            trajectory = nactions                                       # (B, T, 7) original
        cond_data = trajectory
        
       
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        


        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        

        loss_dict = {'bc_loss': loss.item()}
        total_loss = loss
        if self.separate_gripper_head:
            start = self.n_obs_steps - 1
            end = start + self.n_action_steps
            assert end <= horizon, f"gripper window {start}:{end} exceeds horizon {horizon}"

            if self.gripper_condition_on_prev:
                prev_idx = max(start - 1, 0)
                prev_gripper = gripper_target[:, prev_idx]             # (B,) in {0,1}
                if self.training and self.gripper_prev_dropout > 0:
                    drop = torch.rand_like(prev_gripper) < self.gripper_prev_dropout
                    prev_gripper = torch.where(drop, torch.full_like(prev_gripper, 0.5), prev_gripper)
                curr_width = nobs['agent_pos'][:, self.n_obs_steps - 1, self.gripper_width_obs_idx]
                head_in = self._gripper_head_input(global_cond, prev_gripper, curr_width)
            else:
                head_in = global_cond

            gripper_logits = self.gripper_head(head_in)                # (B, n_action_steps)
            g_tgt = gripper_target[:, start:end]
            gripper_loss = F.binary_cross_entropy_with_logits(
                gripper_logits, g_tgt, pos_weight=self.gripper_pos_weight)
            total_loss = loss + self.lambda_gripper * gripper_loss
            loss_dict['gripper_bce'] = gripper_loss.item()

        return total_loss, loss_dict
