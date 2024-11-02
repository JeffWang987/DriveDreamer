import torch

control_model_names = []


def add_control_model_name(model_name):
    model_names = model_name if isinstance(model_name, list) else [model_name]
    for model_name in model_names:
        if model_name not in control_model_names:
            control_model_names.append(model_name)


def get_control_model_names():
    return control_model_names


def repeat_data(data, batch_size=1, num_images_per_prompt=1, num_frames=None, do_classifier_free_guidance=False):
    if num_frames is not None:
        assert num_images_per_prompt == 1
        total_size = batch_size * num_images_per_prompt * num_frames
    else:
        total_size = batch_size * num_images_per_prompt
    if data.shape[0] == 1:
        repeat_by = total_size
    else:
        repeat_by = num_images_per_prompt
    if repeat_by != 1:
        data = data.repeat_interleave(repeat_by, dim=0)
    if do_classifier_free_guidance:
        data = torch.cat([data] * 2)
    return data
