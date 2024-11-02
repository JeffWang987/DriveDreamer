import os

from dreamer_models import load_pipeline

from dreamer_datasets import BaseProcessor, LmdbWriter, load_dataset


class PipelineProcessor(BaseProcessor):
    def __init__(self, pipeline_name, device, save_path, data_name, **kwargs):
        self.pipe = load_pipeline(pipeline_name, lazy=True).to(device)
        self.writer = LmdbWriter(save_path)
        self.data_name = data_name
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data_index = data_dict['data_index']
        image = data_dict['image']
        output_image = self.pipe(image, **self.kwargs)
        return data_index, output_image

    def process(self, data_index, output_image):
        self.writer.write_image(data_index, output_image)

    def close(self):
        self.writer.write_config(data_name=self.data_name)
        self.writer.close()


class LanePipelineProcessor(BaseProcessor):
    def __init__(self, save_path):
        self.pipe = load_pipeline('lane_detection/laneaf/dla34_640x288_batch2_v023', lazy=True)
        self.writer = LmdbWriter(save_path)

    def __call__(self, data_dict):
        data_index = data_dict['data_index']
        image = data_dict['image']
        lanes, lane_labels = self.pipe(image)
        data_dict = {
            'lanes': lanes,
            'lane_labels': lane_labels,
        }
        return data_index, data_dict

    def process(self, data_index, data_dict):
        self.writer.write_dict(data_index, data_dict)

    def close(self):
        self.writer.write_config()
        self.writer.close()


def main():
    data_path = './data/v0.0.1/'
    dataset = load_dataset(data_path)
    processor = PipelineProcessor(
        pipeline_name='edge_detection/canny',
        device='cuda:0',
        save_path=os.path.join(data_path, 'cannys'),
        data_name='image_canny',
        with_attrs=True,
    )
    # processor = LanePipelineProcessor(
    #     save_path=os.path.join(data_path, 'lanes'),
    # )
    dataset.process(processor, num_workers=16)


if __name__ == '__main__':
    main()
