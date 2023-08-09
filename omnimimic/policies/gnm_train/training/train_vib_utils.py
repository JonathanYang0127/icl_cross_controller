import wandb
import os
import numpy as np
from typing import List, Optional, Dict
from prettytable import PrettyTable

from gnm_train.visualizing.action_utils import visualize_action_distribution
from gnm_train.visualizing.distance_utils import visualize_dist_pred
from gnm_train.visualizing.visualize_utils import to_numpy
from gnm_train.training.logger import Logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.distributions.multivariate_normal import MultivariateNormal


def train_vib(
    model: nn.Module,
    optimizer: Adam,
    train_dist_loader: DataLoader,
    train_action_loader: DataLoader,
    device: torch.device,
    project_folder: str,
    epoch: int,
    alpha: float = 0.5,
    beta: float = 0.1,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        train_dist_loader: dataloader for distance training
        train_action_loader: dataloader for action training
        device: device to use
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        beta: weight of kld loss
        learn_angle: whether to learn the angle of the action
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    model.train()
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    kld_logger = Logger("kld", "train", window_size=print_log_freq)

    r = MultivariateNormal(
        torch.zeros(model.distro_size).to(device),
        torch.eye(model.distro_size).to(device),
    )

    variables = [
        dist_loss_logger,
        action_loss_logger,
        action_waypts_cos_sim_logger,
        multi_action_waypts_cos_sim_logger,
        total_loss_logger,
        kld_logger,
    ]

    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "action_orien_cos_sim", "train", window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "multi_action_orien_cos_sim", "train", window_size=print_log_freq
        )
        variables.extend(
            [action_orien_cos_sim_logger, multi_action_orien_cos_sim_logger]
        )

    num_batches = min(len(train_dist_loader), len(train_action_loader))

    for i, val in enumerate(zip(train_dist_loader, train_action_loader)):
        dist_vals, action_vals = val
        (
            dist_obs_image,
            dist_goal_image,
            dist_trans_obs_image,
            dist_trans_goal_image,
            dist_label,
            dist_dataset_index,
        ) = dist_vals
        (
            action_obs_image,
            action_goal_image,
            action_trans_obs_image,
            action_trans_goal_image,
            action_goal_pos,
            action_label,
            action_dataset_index,
        ) = action_vals

        optimizer.zero_grad()

        dist_obs_data = dist_trans_obs_image.to(device)
        dist_goal_data = dist_trans_goal_image.to(device)
        dist_label = dist_label.to(device)
        dist_obs_encoding, dist_z_distro = model.vib_encode(dist_obs_data, dist_goal_data)
        dist_pred, _ = model(dist_obs_encoding, dist_z_distro)

        action_obs_data = action_trans_obs_image.to(device)
        action_goal_data = action_trans_goal_image.to(device)
        action_label = action_label.to(device)
        action_obs_encoding, action_z_distro = model.vib_encode(
            action_obs_data, action_goal_data
        )
        _, action_pred = model(action_obs_encoding, action_z_distro)

        dist_loss, action_loss, kld = model.loss(
            dist_pred, action_pred, dist_label, action_label, dist_z_distro, action_z_distro, r.expand(torch.Size([action_obs_image.shape[0]])),
        )
        total_loss = get_total_loss(dist_loss, action_loss, 0, alpha) + beta * kld
        total_loss.backward()
        optimizer.step()

        # logging stuff
        action_waypts_cos_similairity = F.cosine_similarity(
            action_pred[:, :, :2], action_label[:, :, :2], dim=-1
        ).mean()
        multi_action_waypts_cos_sim = F.cosine_similarity(
            torch.flatten(action_pred[:, :, :2], start_dim=1),
            torch.flatten(action_label[:, :, :2], start_dim=1),
            dim=-1,
        ).mean()
        if learn_angle:
            action_orien_cos_sim = F.cosine_similarity(
                action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
            ).mean()
            multi_action_orien_cos_sim = F.cosine_similarity(
                torch.flatten(action_pred[:, :, 2:], start_dim=1),
                torch.flatten(action_label[:, :, 2:], start_dim=1),
                dim=-1,
            ).mean()
            action_orien_cos_sim_logger.log_data(action_orien_cos_sim.item())
            multi_action_orien_cos_sim_logger.log_data(
                multi_action_orien_cos_sim.item()
            )

        dist_loss_logger.log_data(dist_loss.item())
        action_loss_logger.log_data(action_loss.item())
        action_waypts_cos_sim_logger.log_data(action_waypts_cos_similairity.item())
        multi_action_waypts_cos_sim_logger.log_data(multi_action_waypts_cos_sim.item())
        total_loss_logger.log_data(total_loss.item())
        kld_logger.log_data(kld.item())

        if use_wandb:
            data_log = {}
            for var in variables:
                data_log[var.full_name()] = var.latest()
            wandb.log(data_log)

        if print_log_freq != 0 and  i % print_log_freq == 0:
            log_display = f"(epoch {epoch}) (batch {i}/{num_batches - 1}) "
            for var in variables:
                print(log_display + var.display())
            print()

        if image_log_freq != 0 and i % image_log_freq == 0 and num_images_log > 0:
            visualize_dist_pred(
                to_numpy(dist_obs_image),
                to_numpy(dist_goal_image),
                to_numpy(dist_pred),
                to_numpy(dist_label),
                "train",
                project_folder,
                epoch,
                num_images_log,
                use_wandb=use_wandb,
            )
            visualize_action_distribution(
                model,
                action_obs_image,
                action_goal_image,
                action_obs_encoding,
                action_z_distro,
                r.expand(torch.Size([action_obs_image.shape[0]])),
                action_label,
                action_goal_pos,
                "train",
                project_folder,
                epoch,
                num_images_log,
                use_wandb=use_wandb,
            )
    return


def evaluate_vib(
    eval_type: str,
    model: nn.Module,
    eval_dist_loader: DataLoader,
    eval_action_loader: DataLoader,
    device: torch.device,
    project_folder: str,
    epoch: int = 0,
    alpha: float = 0.5,
    beta: float = 0.1,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        model (nn.Module): model to evaluate
        eval_dist_loader (DataLoader): dataloader for distance prediction
        eval_action_loader (DataLoader): dataloader for action prediction
        device (torch.device): device to use for evaluation
        project_folder (string): path to project folder
        epoch (int): current epoch
        alpha (float): weight for action loss
        beta (float): weight for kld loss
        learn_angle (bool): whether to learn the angle of the action
        print_log_freq (int): frequency of printing loss
        image_log_freq (int): frequency of logging images
        num_images_log (int): number of images to log
        use_wandb (bool): whether to use wandb for logging
        eval_fraction (float): fraction of data to use for evaluation
    """
    model.eval()
    dist_loss_logger = Logger("dist_loss", eval_type, window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", eval_type, window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    total_loss_logger = Logger(
        "total_loss_logger", eval_type, window_size=print_log_freq
    )
    kld_logger = Logger("kld", eval_type, window_size=print_log_freq)

    variables = [
        # dist_loss_logger,
        # action_loss_logger,
        action_waypts_cos_sim_logger,
        multi_action_waypts_cos_sim_logger,
        # total_loss_logger,
    ]
    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "action_orien_cos_sim", eval_type, window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "multi_action_orien_cos_sim", eval_type, window_size=print_log_freq
        )
        variables.extend(
            [action_orien_cos_sim_logger, multi_action_orien_cos_sim_logger]
        )

    num_batches = min(len(eval_dist_loader), len(eval_action_loader))
    num_batches = int(num_batches * eval_fraction) # only use a fraction of the data

    r = MultivariateNormal(
        torch.zeros(model.distro_size).to(device),
        torch.eye(model.distro_size).to(device),
    )

    with torch.no_grad():
        for i, val in enumerate(zip(eval_dist_loader, eval_action_loader)):
            dist_vals, action_vals = val
            (
                dist_obs_image,
                dist_goal_image,
                dist_trans_obs_image,
                dist_trans_goal_image,
                dist_label,
                dist_dataset_index,
            ) = dist_vals
            (
                action_obs_image,
                action_goal_image,
                action_trans_obs_image,
                action_trans_goal_image,
                action_goal_pos,
                action_label,
                action_dataset_index,
            ) = action_vals
            dist_obs_data = dist_trans_obs_image.to(device)
            dist_goal_data = dist_trans_goal_image.to(device)
            dist_label = dist_label.to(device)
            dist_obs_encoding, dist_z_distro = model.vib_encode(dist_obs_data, dist_goal_data)
            dist_pred, _ = model(dist_obs_encoding, dist_z_distro)

            action_obs_data = action_trans_obs_image.to(device)
            action_goal_data = action_trans_goal_image.to(device)
            action_label = action_label.to(device)
            action_obs_encoding, action_z_distro = model.vib_encode(
                action_obs_data, action_goal_data
            )
            _, action_pred = model(action_obs_encoding, action_z_distro)

            dist_loss, action_loss, kld = model.loss(
                dist_pred,
                action_pred,
                dist_label,
                action_label,
                dist_z_distro,
                action_z_distro,
                r.expand(torch.Size([action_obs_image.shape[0]])),
            )
            total_loss = get_total_loss(dist_loss, action_loss, 0, alpha) + beta * kld

            action_waypts_cos_sim = F.cosine_similarity(
                action_pred[:, :, :2], action_label[:, :, :2], dim=-1
            ).mean()
            multi_action_waypts_cos_sim = F.cosine_similarity(
                torch.flatten(action_pred[:, :, :2], start_dim=1),
                torch.flatten(action_label[:, :, :2], start_dim=1),
                dim=-1,
            ).mean()
            if learn_angle:
                action_orien_cos_sim = F.cosine_similarity(
                    action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
                ).mean()
                multi_action_orien_cos_sim = F.cosine_similarity(
                    torch.flatten(action_pred[:, :, 2:], start_dim=1),
                    torch.flatten(action_label[:, :, 2:], start_dim=1),
                    dim=-1,
                ).mean()
                action_orien_cos_sim_logger.log_data(action_orien_cos_sim.item())
                multi_action_orien_cos_sim_logger.log_data(
                    multi_action_orien_cos_sim.item()
                )

            total_loss = alpha * (1e-3 * dist_loss) + (1 - alpha) * action_loss

            dist_loss_logger.log_data(dist_loss.item())
            action_loss_logger.log_data(action_loss.item())
            action_waypts_cos_sim_logger.log_data(action_waypts_cos_sim.item())
            multi_action_waypts_cos_sim_logger.log_data(
                multi_action_waypts_cos_sim.item()
            )
            total_loss_logger.log_data(total_loss.item())

            if print_log_freq != 0 and i % print_log_freq == 0:
                log_display = f"(epoch {epoch}) (batch {i}/{num_batches - 1}) "
                for var in variables:
                    print(log_display + var.display())
                print()

            if image_log_freq != 0 and  i % image_log_freq == 0 and num_images_log > 0:
                visualize_dist_pred(
                    to_numpy(dist_obs_image),
                    to_numpy(dist_goal_image),
                    to_numpy(dist_pred),
                    to_numpy(dist_label),
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    use_wandb=use_wandb,
                )
                visualize_action_distribution(
                    model,
                    action_obs_image,
                    action_goal_image,
                    action_obs_encoding,
                    action_z_distro,
                    r.expand(torch.Size([action_obs_image.shape[0]])),
                    action_label,
                    action_goal_pos,
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    use_wandb=use_wandb,
                )   
            if i >= num_batches - 1: # only use a fraction of the data
                break
    data_log = {}
    for var in variables:
        log_display = f"(epoch {epoch}) "
        data_log[var.full_name()] = var.average()
        print(log_display + var.display())
    print()
    if use_wandb:
        wandb.log(data_log)
    return dist_loss_logger.average(), action_loss_logger.average(), kld_logger.average(), total_loss_logger.average()


def load_model(model, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    loaded_model = checkpoint["model"]
    try:  # for DataParallel
        state_dict = loaded_model.module.state_dict()
        model.load_state_dict(state_dict)
    except (RuntimeError, AttributeError) as e:
        state_dict = loaded_model.state_dict()
        model.load_state_dict(state_dict)


    