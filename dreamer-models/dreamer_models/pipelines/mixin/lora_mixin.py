from ...models import LoRAModel


class LoraMixin:
    def load_lora_model(self, pretrained_model_path, models, scale=1.0):
        load_lora(pretrained_model_path, models, scale)


def load_lora(pretrained_model_path, models, scale=1.0):
    if not isinstance(models, list):
        models = [models]
    lora = LoRAModel.from_pretrained(
        pretrained_model_path,
        torch_dtype=models[0].dtype,
    )
    lora.to(models[0].device)
    lora.to_models(models, scale=scale, is_train=False)
