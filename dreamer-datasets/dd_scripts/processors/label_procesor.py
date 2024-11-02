import numpy as np
from dreamer_models import utils as gm_utils

from dreamer_datasets import BaseProcessor, CLIPTransform, LmdbWriter, load_dataset


class LabelCLIPProcessor(BaseProcessor):
    def __init__(self, save_path, device='cuda'):
        model_path = gm_utils.get_model_path('openai/clip-vit-large-patch14')
        self.transform = CLIPTransform(model_path, device=device)
        self.writer = LmdbWriter(save_path)

    def __call__(self, data_dict):
        data_index = data_dict['data_index']
        labels = data_dict['labels']
        if len(labels) > 0:
            label_embeds = []
            for j in range(len(labels)):
                label_embeds.append(self.transform(labels[j], text_w_proj=False))
        else:
            self.transform.load_model()
            label_embeds = np.zeros((0, self.transform.model.text_embed_dim), dtype=np.float32)
        return data_index, {'label_embeds': label_embeds}

    def process(self, data_index, label_embeds):
        self.writer.write_dict(data_index, label_embeds)

    def close(self):
        self.writer.write_config()
        self.writer.close()


def main():
    data_path = './labels/'
    save_path = './label_embeds/'
    dataset = load_dataset(data_path)
    processor = LabelCLIPProcessor(save_path=save_path, device='cuda:0')
    dataset.process(processor, num_workers=8)


if __name__ == '__main__':
    main()
