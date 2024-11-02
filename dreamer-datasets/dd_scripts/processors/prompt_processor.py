from dreamer_inference.text.clip_interrogator import ClipInterrogator
from dreamer_models import utils as gm_utils

from dreamer_datasets import BaseProcessor, CLIPTextTransform, CLIPTextWithProjectionTransform, LmdbWriter, load_dataset


class PromptSDProcessor(BaseProcessor):
    def __init__(self, save_path, device='cuda'):
        model_path = gm_utils.get_model_path('runwayml/stable-diffusion-v1-5')
        self.clip_model = ClipInterrogator(device)
        self.transform = CLIPTextTransform(
            model_path=model_path,
            device=device,
            dtype='float16',
        )
        self.writer = LmdbWriter(save_path)
        self.default_prompt = 'realistic autonomous driving scene'

    def __call__(self, data_dict):
        data_index = data_dict['data_index']
        if 'prompts' in data_dict:
            prompt_embeds = self.transform(data_dict['prompts'], mode='before_pool')
            new_data_dict = {'prompt_embeds': prompt_embeds}
        else:
            image = data_dict['image']
            prompts = self.clip_model.inference(image, modes=['fast_prompt'])
            if self.default_prompt is not None:
                for i in range(len(prompts)):
                    prompts[i] = ', '.join([self.default_prompt, prompts[i]])
            prompt_embeds = self.transform(prompts, mode='before_pool')
            new_data_dict = {
                'prompts': prompts,
                'prompt_embeds': prompt_embeds,
            }
        return data_index, new_data_dict

    def process(self, data_index, data_dict):
        self.writer.write_dict(data_index, data_dict)

    def close(self):
        self.writer.write_config()
        self.writer.close()


class PromptSDXLProcessor(BaseProcessor):
    def __init__(self, save_path, device='cuda'):
        model_path = gm_utils.get_model_path('stabilityai/stable-diffusion-xl-base-1.0')
        self.transform = CLIPTextWithProjectionTransform(
            model_path=model_path,
            device=device,
            dtype='float16',
        )
        self.writer = LmdbWriter(save_path)
        self.default_prompt = 'realistic autonomous driving scene'

    def __call__(self, data_dict):
        data_index = data_dict['data_index']
        if 'prompts' in data_dict:
            prompt_embeds = self.transform(data_dict['prompts'], mode='before_pool')
            new_data_dict = {'prompt_embeds': prompt_embeds[0], 'pooled_prompt_embeds': prompt_embeds[1]}
        else:
            image = data_dict['image']
            prompts = self.clip_model.inference(image, modes=['fast_prompt'])
            if self.default_prompt is not None:
                for i in range(len(prompts)):
                    prompts[i] = ', '.join([self.default_prompt, prompts[i]])
            prompt_embeds = self.transform(prompts, mode='before_pool')
            new_data_dict = {
                'prompts': prompts,
                'prompt_embeds': prompt_embeds[0],
                'pooled_prompt_embeds': prompt_embeds[1],
            }
        return data_index, new_data_dict

    def process(self, data_index, data_dict):
        self.writer.write_dict(data_index, data_dict)

    def close(self):
        self.writer.write_config()
        self.writer.close()


def main():
    data_path = './labels/'
    save_path = './prompt_embeds/'
    dataset = load_dataset(data_path)
    processor = PromptSDProcessor(save_path=save_path, device='cuda:0')
    dataset.process(processor, num_workers=8)


if __name__ == '__main__':
    main()
