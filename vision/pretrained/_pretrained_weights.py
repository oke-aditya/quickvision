import torch

__all__ = ["_load_pretrained_weights"]


def _load_pretrained_weights(weights_dict, model, model_name: str, pretrained: str):
    state_dict = torch.hub.load_state_dict_from_url(weights_dict[model_name][pretrained], map_location="cpu")
    model.load_state_dict(state_dict)
    return model
