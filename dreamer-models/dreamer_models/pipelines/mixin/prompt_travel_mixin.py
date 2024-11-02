import torch


class PromptTravelMixin:
    def process_prompt_embeds(self, prompt_embeds, prompt, num_frames=None):
        if num_frames is not None and isinstance(prompt, list) and len(prompt) == num_frames:
            prompt_map = get_prompt_map(prompt)
            new_prompt_embeds = []
            for i in range(len(prompt_embeds)):
                key_prev = list(prompt_map.keys())[0]
                key_next = list(prompt_map.keys())[-1]
                center_frame = i
                for p in prompt_map.keys():
                    if p > center_frame:
                        key_next = p
                        break
                    key_prev = p
                dist_prev = center_frame - key_prev
                if dist_prev < 0:
                    dist_prev += num_frames
                dist_next = key_next - center_frame
                if dist_next < 0:
                    dist_next += num_frames
                if key_prev == key_next or dist_prev + dist_next == 0:
                    new_prompt_embeds.append(prompt_embeds[key_prev])
                else:
                    rate = dist_prev / (dist_prev + dist_next)
                    new_prompt_embeds.append(slerp(prompt_embeds[key_prev], prompt_embeds[key_next], rate))
            prompt_embeds = torch.stack(new_prompt_embeds).to(prompt_embeds.dtype).to(prompt_embeds.device)
        return prompt_embeds


def get_prompt_map(prompt):
    prompt_map = dict()
    prompt_map[0] = prompt[0]
    prev_prompt = prompt[0]
    for idx in range(1, len(prompt)):
        if prompt[idx] != prev_prompt:
            prompt_map[idx] = prompt[idx]
            prev_prompt = prompt[idx]
    return prompt_map


def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        return (1.0 - t) * v0 + t * v1
    omega = dot.acos()
    return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()
