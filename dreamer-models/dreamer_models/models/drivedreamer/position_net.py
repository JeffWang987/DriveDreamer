import torch
import torch.nn as nn

from .convnext import convnext_tiny


def get_fourier_embeds_from_boundingbox(embed_dim, box):
    """
    Args:
        embed_dim: int
        box: a 3-D tensor [B x N x M] representing the bounding boxes
    Returns:
        [B x N x embed_dim] tensor of positional embeddings
    """

    batch_size, num_boxes, box_dim = box.shape

    emb = 100 ** (torch.arange(embed_dim) / embed_dim)
    emb = emb[None, None, None].to(device=box.device, dtype=box.dtype)
    emb = emb * box.unsqueeze(-1)

    emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
    emb = emb.permute(0, 1, 3, 4, 2).reshape(batch_size, num_boxes, embed_dim * 2 * box_dim)

    return emb


class PositionNet(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, box_dim=4, feature_type='text_only', fourier_freqs=8):
        super().__init__()
        position_dim = fourier_freqs * 2 * box_dim  # 2: sin/cos
        self.feature_type = feature_type
        self.fourier_embedder_dim = fourier_freqs
        self.null_position_feature = nn.Parameter(torch.zeros([position_dim]))

        if feature_type == 'text_only':
            self.linears = nn.Sequential(
                nn.Linear(in_dim + position_dim, mid_dim),
                nn.SiLU(),
                nn.Linear(mid_dim, mid_dim),
                nn.SiLU(),
                nn.Linear(mid_dim, out_dim),
            )
            self.null_positive_feature = nn.Parameter(torch.zeros([in_dim]))

        elif feature_type == 'text_image':
            self.linears_text = nn.Sequential(
                nn.Linear(in_dim + position_dim, mid_dim),
                nn.SiLU(),
                nn.Linear(mid_dim, mid_dim),
                nn.SiLU(),
                nn.Linear(mid_dim, out_dim),
            )
            self.linears_image = nn.Sequential(
                nn.Linear(in_dim + position_dim, mid_dim),
                nn.SiLU(),
                nn.Linear(mid_dim, mid_dim),
                nn.SiLU(),
                nn.Linear(mid_dim, out_dim),
            )
            self.null_text_feature = nn.Parameter(torch.zeros([in_dim]))
            self.null_image_feature = nn.Parameter(torch.zeros([in_dim]))

        else:
            assert False

    def forward(
        self,
        boxes,
        masks,
        positive_embeddings=None,
        text_embeddings=None,
        text_masks=None,
        image_embeddings=None,
        image_masks=None,
    ):
        masks = masks.unsqueeze(-1)
        xyxy_embedding = get_fourier_embeds_from_boundingbox(self.fourier_embedder_dim, boxes)
        xyxy_null = self.null_position_feature.view(1, 1, -1)
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        if self.feature_type == 'text_only':
            positive_null = self.null_positive_feature.view(1, 1, -1)
            positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null
            objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1))

        elif self.feature_type == 'text_image':
            text_masks = text_masks.unsqueeze(-1)
            image_masks = image_masks.unsqueeze(-1)
            text_null = self.null_text_feature.view(1, 1, -1)
            image_null = self.null_image_feature.view(1, 1, -1)
            text_embeddings = text_embeddings * text_masks + (1 - text_masks) * text_null
            image_embeddings = image_embeddings * image_masks + (1 - image_masks) * image_null
            objs_text = self.linears_text(torch.cat([text_embeddings, xyxy_embedding], dim=-1))
            objs_image = self.linears_image(torch.cat([image_embeddings, xyxy_embedding], dim=-1))
            objs = torch.cat([objs_text, objs_image], dim=1)

        else:
            assert False

        return objs


class ImagePositionNet(nn.Module):
    def __init__(self, resize_input=448, mid_dim=512, out_dim=768):
        super().__init__()
        self.resize_input = resize_input
        self.down_factor = 32  # determined by the convnext backbone
        self.out_dim = out_dim
        assert self.resize_input % self.down_factor == 0
        self.convnext_tiny_backbone = convnext_tiny(pretrained=True)
        self.num_tokens = (self.resize_input // self.down_factor) ** 2
        convnext_feature_dim = 768
        self.pos_embedding = nn.Parameter(torch.empty(1, self.num_tokens, convnext_feature_dim).normal_(std=0.02))
        self.linears = nn.Sequential(
            nn.Linear(convnext_feature_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, image):
        # token from image map
        image = torch.nn.functional.interpolate(image, self.resize_input)
        image_feature = self.convnext_tiny_backbone(image)
        objs = image_feature.reshape(image_feature.shape[0], -1, self.num_tokens)
        objs = objs.permute(0, 2, 1).contiguous()  # N*Num_tokens*dim
        # add pos
        objs = objs + self.pos_embedding
        # fuse them
        objs = self.linears(objs)
        return objs
