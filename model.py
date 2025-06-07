import torch 
import torch.nn as nn
from clip import clip
from transformer import *
import torch.nn.functional as F
from transformers.models.t5.modeling_t5 import T5Block, T5Stack, T5LayerCrossAttention 
from transformers.models.t5 import T5Config
from BLIP.models.blip_retrieval import blip_retrieval

MODEL_PATH = "/home/hle/spinning-storage/hle/ckpt" 

class TransAgg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        self.type = cfg.type 
        self.model_name = cfg.model
        if self.model_name == 'blip':
            self.model = blip_retrieval(pretrained=f"{MODEL_PATH}/model_base_retrieval_coco.pth")
            self.feature_dim = 256
        elif self.model_name == "blip_flickr":
            self.model = blip_retrieval(pretrained=f"{MODEL_PATH}/model_base_retrieval_flickr.pth")
            self.feature_dim = 256
        elif self.model_name == "freeblip":
            self.model = blip_retrieval(pretrained=f"{MODEL_PATH}/FreeBLIP.pth")#, image_size=224, vit='large', vit_grad_ckpt=True, vit_ckpt_layer=10)
            self.feature_dim = 256
        elif self.model_name == 'clip_base':
            self.model, self.preprocess = clip.load(f"{MODEL_PATH}/ViT-B-32.pt", device=cfg.device, jit=False)
            self.feature_dim = self.model.visual.output_dim 
        elif self.model_name == 'clip_large':
            self.model, self.preprocess = clip.load(f"{MODEL_PATH}/ViT-L-14.pt", device=cfg.device, jit=False)
            self.feature_dim = self.model.visual.output_dim 
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=8, dropout=cfg.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.logit_scale = 100
        self.dropout = nn.Dropout(cfg.dropout)
        self.combiner_layer = nn.Linear(self.feature_dim + self.feature_dim, (self.feature_dim + self.feature_dim) * 4)
        self.weighted_layer = nn.Linear(self.feature_dim, 3)
        self.output_layer = nn.Linear((self.feature_dim + self.feature_dim) * 4, self.feature_dim)
        self.sep_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))

        self.union_weighted_layer = nn.Sequential(
                nn.Linear(self.feature_dim * 2, self.feature_dim * 4), 
                nn.Dropout(cfg.dropout/2), 
                nn.Sigmoid(),
                nn.Linear(self.feature_dim * 4, self.feature_dim), 
                nn.Dropout(cfg.dropout/2), 
                nn.Sigmoid(), 
                nn.Linear(self.feature_dim, 1))
        # T5 layers for feature fusion
        if cfg.model in ["clip_base", "blip", "blip_flickr"]:
            conf_t5 = T5Config()
            conf_t5.num_layers = 4
            conf_t5.num_decoder_layers = 4
            conf_t5.num_heads = 8
            conf_t5.d_model = self.feature_dim 
            conf_t5.d_kv = 64
            conf_t5.feed_forward_proj = "relu"
            self.t5_layers = T5Stack(conf_t5)
        elif cfg.model == "clip_large" or cfg.model == "freeblip":
            conf_t5_vit_large = T5Config()
            conf_t5_vit_large.num_layers = 4
            conf_t5_vit_large.num_decoder_layers = 4
            conf_t5_vit_large.num_heads = 12
            conf_t5_vit_large.d_model = self.feature_dim
            conf_t5_vit_large.d_kv = 64
            conf_t5_vit_large.feed_forward_proj = "relu"
            self.t5_layers = T5Stack(conf_t5_vit_large)
        else:
            raise NotImplementedError("Only ViT-B/32, ViT-L/14 and BLIP are supported.")


    def forward(self, texts, reference_images, target_images, reference_captions = None, target_captions = None):
        img_text_rep = self.combine_features(reference_images, texts, reference_captions)
        null_texts = [""] * len(target_images) 
        if self.model_name.startswith('blip'):
            tokenized_null_texts = self.model.tokenizer(null_texts, padding='max_length', truncation=True, max_length=35,
                                                              return_tensors='pt').to(self.device)
        elif self.model_name.startswith('clip'):
            tokenized_null_texts = clip.tokenize(null_texts, truncate=True).to(self.device)
        text_target_features, _ = self.model.encode_text(tokenized_null_texts)
        target_features, _ = self.model.encode_image(target_images)
        target_features = F.normalize(target_features, dim=-1)
        if self.type == "sum":
            target_features += F.normalize(text_target_features, dim = -1)
        elif self.type == "union":
            target_features = self.union_features(target_images, null_texts)
        
        union_features = self.union_features(reference_images, texts)
        logits = self.logit_scale * (img_text_rep @ target_features.T) #+ self.logit_scale * (union_features @ target_features.T)
        
        return logits

    def union_features(self, reference_images, texts):
        img_embeds, i_e_total = self.model.encode_image(reference_images, return_local = True)
        
        if self.model_name.startswith('blip'):
            tokenized_texts = self.model.tokenizer(texts, padding='max_length', truncation=True, max_length=35,
                    return_tensors='pt').to(self.device)
        elif self.model_name.startswith('clip'):
            tokenized_texts = clip.tokenize(texts, truncate = True).to(self.device)
        txt_embeds, t_e_total = self.model.encode_text(tokenized_texts) 
        sum_embeds = F.normalize(img_embeds + txt_embeds, dim =-1)
        concat = torch.cat((img_embeds.unsqueeze(1), txt_embeds.unsqueeze(1)), dim=1)
        transformer_embed = self.t5_layers(
                inputs_embeds = concat, 
                attention_mask = None,
                use_cache = False,
                return_dict = True
                )
        concat = transformer_embed.last_hidden_state
        i_w, t_w = concat[:, 0] , concat[:, 1]
        #concat = F.normalize(concat, dim = -1)
        union_feats = torch.cat((i_w, t_w), dim = -1)
        union_weighted = self.union_weighted_layer(union_feats) 
        output_rep = union_weighted[:, 0:1] * img_embeds + (1 - union_weighted[:, 0:1]) * txt_embeds
        output_rep = F.normalize(output_rep, dim = -1)
        return output_rep

    
    def combine_features(self, reference_images, texts, reference_captions = None):
        reference_image_features, reference_total_image_features = self.model.encode_image(reference_images, return_local=True)
        
        batch_size = reference_image_features.size(0)
        reference_total_image_features = reference_total_image_features.float()
        
        if self.model_name.startswith('blip'):
            tokenized_texts = self.model.tokenizer(texts, padding='max_length', truncation=True, max_length=35, return_tensors='pt').to(self.device)
            mask = (tokenized_texts.attention_mask == 0)
        elif self.model_name.startswith('clip'):
            tokenized_texts = clip.tokenize(texts, truncate=True).to(self.device)
            mask = (tokenized_texts == 0)
        text_features, total_text_features = self.model.encode_text(tokenized_texts)

        union_features = self.union_features(reference_images, texts)

        num_patches = reference_total_image_features.size(1)
        sep_token = self.sep_token.repeat(batch_size, 1, 1)

        combine_features = torch.cat((total_text_features, sep_token, reference_total_image_features), dim=1)

        image_mask = torch.zeros(batch_size, num_patches + 1).to(reference_image_features.device)
        mask = torch.cat((mask, image_mask), dim=1)
        
        img_text_rep = self.fusion(combine_features, src_key_padding_mask=mask) 
        
        if self.model_name.startswith('blip'): 
            multimodal_img_rep = img_text_rep[:, 36, :] 
            multimodal_text_rep = img_text_rep[:, 0, :]
        elif self.model_name.startswith('clip'):
            multimodal_img_rep = img_text_rep[:, 78, :]
            multimodal_text_rep = img_text_rep[torch.arange(batch_size), tokenized_texts.argmax(dim=-1), :]

        concate = torch.cat((multimodal_img_rep, multimodal_text_rep), dim=-1)
        f_U = self.output_layer(self.dropout(F.relu(self.combiner_layer(concate))))
        weighted = self.weighted_layer(f_U) # (batch_size, 3)
        
        query_rep = weighted[:, 0:1] * text_features + weighted[:, 1:2] * f_U + weighted[:, 2:3] * reference_image_features
        
        query_rep = F.normalize(query_rep, dim=-1)

        return query_rep 
