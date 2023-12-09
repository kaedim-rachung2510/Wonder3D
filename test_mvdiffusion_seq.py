import argparse
import os
from typing import Dict, Optional, List
from PIL import Image
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.utils.checkpoint
from torchvision.utils import save_image

from accelerate.utils import set_seed
from tqdm.auto import tqdm
from einops import rearrange
from rembg import remove

from Wonder3D.mvdiffusion.data.single_image_dataset import SingleImageDataset as MVDiffusionDataset
from helpers.models import HallucinationsPipeline

@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path:str
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int

    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str  # joint, or ablation

    enable_xformers_memory_efficient_attention: bool

    cond_on_normals: bool
    cond_on_colors: bool

def log_validation(dataloader, pipeline, generator, cfg: TestConfig, name, save_dir):

    VIEWS = get_views(cfg.num_views)
    
    images_cond, images_pred = [], defaultdict(list)
    for i, batch in tqdm(enumerate(dataloader)):
        # (B, Nv, 3, H, W)
        imgs_in = batch['imgs_in']
        alphas = batch['alphas']
        # (B, Nv, Nce)
        camera_embeddings = batch['camera_embeddings']
        filename = batch['filename']

        bsz, num_views = imgs_in.shape[0], imgs_in.shape[1]
        # (B*Nv, 3, H, W)
        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")
        alphas = rearrange(alphas, "B Nv C H W -> (B Nv) C H W")
        # (B*Nv, Nce)
        camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

        images_cond.append(imgs_in)

        with torch.autocast("cuda"):
            # B*Nv images
            for guidance_scale in cfg.validation_guidance_scales:
                out = pipeline(
                    imgs_in, camera_embeddings, generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
                ).images
                images_pred[f"{name}-sample_cfg{guidance_scale:.1f}"].append(out)
                # cur_dir = os.path.join(save_dir, "hallucinations")

                # pdb.set_trace()
                for i in range(bsz):
                    scene = os.path.basename(filename[i])
                    scene_dir = save_dir
                    outs_dir = os.path.join(scene_dir, "outs")
                    masked_outs_dir = os.path.join(scene_dir, "masked_outs")
                    os.makedirs(outs_dir, exist_ok=True)
                    os.makedirs(masked_outs_dir, exist_ok=True)
                    img_in = imgs_in[i*num_views]
                    alpha = alphas[i*num_views]
                    img_in = torch.cat([img_in, alpha], dim=0)
                    save_image(img_in, os.path.join(scene_dir, scene+".png"))
                    for j in range(num_views):
                        view = VIEWS[j]
                        idx = i*num_views + j
                        pred = out[idx]

                        # pdb.set_trace()
                        out_filename = f"{cfg.pred_type}_000_{view}.png"
                        pred = save_image(pred, os.path.join(outs_dir, out_filename))

                        rm_pred = remove(pred)

                        save_image_numpy(rm_pred, os.path.join(scene_dir, out_filename))
    torch.cuda.empty_cache()

def save_image(tensor, fp=""):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    if fp:
        im.save(fp)
    return ndarr

def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)

def log_validation_joint(dataloader, pipeline, generator, cfg: TestConfig, name, save_dir):

    VIEWS = get_views(cfg.num_views)
    
    images_cond, normals_pred, images_pred = [], defaultdict(list), defaultdict(list)
    for i, batch in tqdm(enumerate(dataloader)):
        # repeat  (2B, Nv, 3, H, W)
        imgs_in = torch.cat([batch['imgs_in']]*2, dim=0)

        filename = batch['filename']
        
        # (2B, Nv, Nce)
        camera_embeddings = torch.cat([batch['camera_embeddings']]*2, dim=0)

        task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0)

        camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

        # (B*Nv, 3, H, W)
        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")
        # (B*Nv, Nce)
        camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

        images_cond.append(imgs_in)
        num_views = len(VIEWS)
        with torch.autocast("cuda"):
            # B*Nv images
            for guidance_scale in cfg.validation_guidance_scales:
                out = pipeline(
                    imgs_in, camera_embeddings, generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
                ).images

                bsz = out.shape[0] // 2
                normals_pred = out[:bsz]
                images_pred = out[bsz:]

                for i in range(bsz//num_views):
                    scene_dir = save_dir
                    for j in range(num_views):
                        view = VIEWS[j]
                        idx = i*num_views + j
                        normal = normals_pred[idx]
                        color = images_pred[idx]

                        normal_filename = f"normals_000_{view}.png"
                        rgb_filename = f"rgb_000_{view}.png"

                        # normal = save_image(normal, os.path.join(normal_dir, normal_filename))
                        # color = save_image(color, os.path.join(scene_dir, rgb_filename))
                        normal = save_image(normal)
                        color = save_image(color)

                        rm_normal = remove(normal)
                        rm_color = remove(color)

                        save_image_numpy(rm_normal, os.path.join(scene_dir, normal_filename))
                        save_image_numpy(rm_color, os.path.join(scene_dir, rgb_filename))

    torch.cuda.empty_cache()

def main(cfg: TestConfig):

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Load pipeline with scheduler, tokenizer and models.
    pipeline, generator = HallucinationsPipeline().load(cfg)

    # Get the dataset
    validation_dataset = MVDiffusionDataset(
        **cfg.validation_dataset
    )

    # DataLoaders creation:
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    )

    os.makedirs(cfg.save_dir, exist_ok=True)

    if cfg.pred_type == 'joint':
        log_validation_joint(
                    validation_dataloader,
                    pipeline,
                    generator,
                    cfg,
                    'validation',
                    cfg.save_dir
                    )
    else:
        log_validation(
                    validation_dataloader,
                    pipeline,
                    generator,
                    cfg,
                    'validation',
                    cfg.save_dir
                    )
    torch.cuda.empty_cache()
    
def get_views(num_views):
    VIEWS = []
    if num_views == 6:
        VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
    elif num_views == 4:
        VIEWS = ['front', 'right', 'back', 'left']
    return VIEWS    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, extras = parser.parse_known_args()

    from utils.misc import load_config
    from omegaconf import OmegaConf

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    # print(cfg)
    schema = OmegaConf.structured(TestConfig)
    # cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)

    if cfg.num_views == 6:
        VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
    elif cfg.num_views == 4:
        VIEWS = ['front', 'right', 'back', 'left']
    main(cfg)
