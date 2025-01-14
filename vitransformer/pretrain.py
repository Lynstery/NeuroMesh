from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.layers import trunc_normal_ as __call_trunc_normal_
from timm.models import register_model

from mesh.vitransformer.finetune import (Block, PatchEmbed, _cfg,
                                         get_sinusoid_encoding_table)

__all__ = [
    'pretrain_videomae_small_patch16_224',
    'pretrain_videomae_base_patch16_224',
    'pretrain_videomae_large_patch16_224',
    'mesh_encode_model'
    'mesh_decode_model'
]


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


COMPRESSED_DATA_DIM = 64


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2,
                 use_learnable_pos_emb=False, mask_ratio=0.9):
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.visible_patches = int(num_patches*(1-mask_ratio))
        #print("No. of visible patches selected for pre-training: {}".format(self.visible_patches))

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(
                num_patches, embed_dim)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Probability prediction network
        self.pos_embed_probs = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim))
        self.get_token_probs = nn.ModuleList([
            Block(dim=embed_dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                  drop=0.1, attn_drop=0.00, drop_path=0.00, norm_layer=nn.LayerNorm,
                  init_values=0.),
            nn.Linear(embed_dim, 1),
            torch.nn.Flatten(start_dim=1)
        ])

        self.softmax = nn.Softmax(dim=-1)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

        self.content_complexity = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def get_mask(self, x):
        # detach()
        x = x + self.pos_embed_probs.type_as(x).to(x.device).clone()
        #logits = self.get_token_probs(x)
        x, _ = self.get_token_probs[0](x)
        x = self.get_token_probs[1](x)
        logits = self.get_token_probs[2](x)

        logits = torch.nan_to_num(logits)
        p_x = self.softmax(logits)
        vis_idx = torch.multinomial(
            p_x, num_samples=self.visible_patches, replacement=False)
        mask = torch.ones((x.shape[0], x.shape[1])).to(
            x.device, non_blocking=True)
        mask.scatter_(dim=-1, index=vis_idx.long(), value=0.0)
        mask = mask.flatten(1).to(torch.bool)
        return p_x, vis_idx, mask

    def forward_features(self, x):
        _, _, T, _, _ = x.shape  # 8, 3, 16(T), 224, 224
        x = self.patch_embed(x)  # 8, 1568 (224/16 x 224/16 x 16/2), 768

        p_x, vis_idx, mask = self.get_mask(x)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        B, _, C = x.shape
        # ~mask means visible shape: 8, 160, 768
        x_vis = x[~mask].reshape(B, -1, C)

        content_complexity_list = []
        for blk in self.blocks:
            x_vis, content_complexity = blk(x_vis)
            content_complexity_list.append(content_complexity)

        content_complexity = torch.mean(torch.stack(content_complexity_list))

        x_vis = self.norm(x_vis)  # 8, 160, 768
        return x_vis, p_x, vis_idx, mask, content_complexity

    def get_p_x(self, x):
        _, _, T, _, _ = x.shape  # 8, 3, 16(T), 224, 224
        x = self.patch_embed(x)  # 8, 1568 (224/16 x 224/16 x 16/2), 768
        p_x, _, _ = self.get_mask(x)
        return p_x

    def forward(self, x):
        x, p_x, vis_idx, mask, content_complexity = self.forward_features(x)
        x = self.head(x)
        return x, p_x, mask, content_complexity


class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2
                 ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        self.content_complexity = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def post_processing_for_tokens(self, x, return_token_num : int):
        if return_token_num > 0:
            # only return the mask tokens predict pixels
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))
        return x

    def forward(self, x, return_token_num: int):
        depth = 0
        multilayer_outputs = []
        content_complexity_list = []
        for blk in self.blocks:
            x, content_complexity = blk(x)
            multilayer_outputs.append(self.post_processing_for_tokens(x, return_token_num))
            content_complexity_list.append(content_complexity)
            depth += 1
        multilayer_outputs = torch.stack(multilayer_outputs)
        centent_complexity = torch.mean(torch.stack(content_complexity_list))
        return multilayer_outputs, centent_complexity


class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 pretrained_cfg=None,
                 pretrained_cfg_overlay=None,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=1536,  # decoder_num_classes=768,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=2,
                 mask_ratio=0.9,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 init_ckpt=None,
                 decoder_eval=True
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            mask_ratio=mask_ratio)

        #self.encoder_output = nn.Linear(
        #    encoder_embed_dim, COMPRESSED_DATA_DIM, bias=False)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size)
        self.decoder_eval = decoder_eval

        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, decoder_undergo_depth=None):
        _, _, T, _, _ = x.shape
        x_vis, p_x, mask = self.encoder(x)  # [B, N_vis, C_e]
        #x_vis = self.encoder_output(x_vis)
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(
            B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat(
            [x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)  # [B, N, C_d]
        # [B, N_mask, 3 * 16 * 16] pos_emd_mask.shape[1]
        # x_mask = self.decoder(x_full, pos_emd_mask.shape[1])
        if decoder_undergo_depth is None:
            decoder_undergo_depth = self.decoder.depth
        multilayer_outputs = self.decoder(
            x_full, -1, decoder_undergo_depth, self.decoder_eval)
        return multilayer_outputs, p_x, mask
    
    def encoder_forward(self, x):
        _, _, T, _, _ = x.shape
        x_vis, p_x, mask = self.encoder(x)  # [B, N_vis, C_e]
        #x_vis = self.encoder_output(x_vis)
        return x_vis, p_x, mask
    
    def decoder_forward(self, x_vis, mask):
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x_vis).to(x_vis.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        # [B, N_mask, 3 * 16 * 16] pos_emd_mask.shape[1]
        # x_mask = self.decoder(x_full, pos_emd_mask.shape[1])
        #if decoder_undergo_depth is None:
        decoder_undergo_depth = self.decoder.depth
        multilayer_outputs = self.decoder(
            x_full, -1, decoder_undergo_depth, self.decoder_eval)
        return multilayer_outputs

class MeshEncoderNet(nn.Module):
    def __init__(self,
                 pretrained_cfg=None,
                 pretrained_cfg_overlay=None,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=1536,  # decoder_num_classes=768,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=2,
                 mask_ratio=0.9,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 state_dict=None,
                 decoder_eval=True
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            mask_ratio=mask_ratio)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        _, _, T, _, _ = x.shape
        x_vis, p_x, mask, content_complexity = self.encoder(x)  # [B, N_vis, C_e]
        return x_vis, p_x, mask, content_complexity 

class MeshDecoderNet(nn.Module):
    def __init__(self,
                 pretrained_cfg=None,
                 pretrained_cfg_overlay=None,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=1536,  # decoder_num_classes=768,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=2,
                 mask_ratio=0.9,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 state_dict=None,
                 decoder_eval=True
                 ):
        
        super().__init__()
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        num_patches = (224 // 16) * (224 // 16) * (16 // 2)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=num_patches, #self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size)

        self.pos_embed = get_sinusoid_encoding_table(num_patches, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_vis, mask):
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x_vis).to(x_vis.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        multilayer_outputs = self.decoder(x_full, -1)
        return multilayer_outputs

@register_model
def pretrain_videomae_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mask_ratio=0.9,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def mesh_encode_model(pretrained=True, mask_ratio=0.9, **kwargs):
    model = MeshEncoderNet(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mask_ratio=mask_ratio,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = kwargs["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    return model

@register_model
def mesh_decode_model(pretrained=True, mask_ratio=0.9, **kwargs):
    model = MeshDecoderNet(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mask_ratio=mask_ratio,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = kwargs["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    return model

@register_model
def pretrain_videomae_base_patch16_224(pretrained=False, mask_ratio=0.9, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mask_ratio=mask_ratio,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_videomae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mask_ratio=0.9,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
    return model
