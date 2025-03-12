from omegaconf import OmegaConf


def show_cfg(cfg):
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(resolved_cfg))
