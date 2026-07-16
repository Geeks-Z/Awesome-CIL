import torch
import torch.nn as nn
import copy

from backbone.vit_bilora import VisionTransformer, PatchEmbed, resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn, Block, Attention_LoRA
import torch.nn.functional as F

class Attention_FFT(Attention_LoRA):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10, n_frq=3000):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, r, n_tasks)

        self.n_frq = n_frq
        self.coef_k = nn.ParameterList([nn.Parameter(torch.randn(self.n_frq), requires_grad=True) for _ in range(n_tasks)]).to(self.qkv.weight.device)
        self.coef_v = nn.ParameterList([nn.Parameter(torch.randn(self.n_frq), requires_grad=True) for _ in range(n_tasks)]).to(self.qkv.weight.device)

        self.indices = [self.select_pos(t, self.dim).to(self.qkv.weight.device) for t in range(n_tasks)]
        self.MoE = False
        if self.MoE:
            self.gate = nn.Linear(self.dim, n_tasks)
           
    def init_param(self):
        for t in range(len(self.coef_k)):
            nn.init.zeros_(self.coef_k[t])
        for t in range(len(self.coef_v)):
            nn.init.zeros_(self.coef_v[t])
    
    def select_pos(self, t, dim, seed=777):
        indices = torch.randperm(dim * dim, generator=torch.Generator().manual_seed(seed+t*10))[:self.n_frq]
        indices = torch.stack([indices // dim, indices % dim], dim=0)
        return indices
    
    def get_delta_w_k(self, task, alpha=300):
        indices = self.indices[task]
        F = torch.zeros(self.dim, self.dim).to(self.qkv.weight.device)
        F[indices[0,:], indices[1,:]] =  self.coef_k[task]
        return torch.fft.ifft2(F, dim=(-2,-1)).real * alpha
    
    def get_delta_w_v(self, task, alpha=300):
        indices = self.indices[task]
        F = torch.zeros(self.dim, self.dim).to(self.qkv.weight.device)
        F[indices[0,:], indices[1,:]] =  self.coef_v[task]
        return torch.fft.ifft2(F, dim=(-2,-1)).real * alpha

    def forward(self, x, task, register_hook=False, get_feat=False,get_cur_feat=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if self.MoE:
            gate_logits = self.gate(x)  # Shape: (batch_size, num_experts)

            mask = torch.zeros(self.n_tasks).to(self.qkv.weight.device)
            mask[:task] = 1
            gate_logits = gate_logits.masked_fill(mask == 0, float('-inf'))

            # Compute softmax over masked logits
            gate_values = F.softmax(gate_logits, dim=-1)  # Shape: (batch_size, num_experts)

            # Compute expert outputs
            expert_outputs = torch.stack([self.get_delta_w(t) for t in range(task+1)], dim=0).sum(dim=0)  # Shape: (batch_size, num_experts, expert_dim)

            # Weighted sum of expert outputs
            weighted_expert_output = torch.einsum('be,bed->bd', gate_values, expert_outputs)  # Shape: (batch_size, expert_dim)

        else: 
            if task > -0.5:
                weight_k = torch.stack([self.get_delta_w_k(t) for t in range(task+1)], dim=0).sum(dim=0)
                weight_v = torch.stack([self.get_delta_w_v(t) for t in range(task+1)], dim=0).sum(dim=0)
        k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block_FFT(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_tasks=10, r=64):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, init_values, drop_path, act_layer, norm_layer, n_tasks, r)
        self.attn = Attention_FFT(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, n_tasks=n_tasks, r=r)

class ViT_lora_fft(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block_FFT, n_tasks=10, rank=64):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, n_tasks=n_tasks, rank=rank)


    def forward(self, x, task_id, register_blk=-1, get_feat=False, get_cur_feat=False):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)
        for i, blk in enumerate(self.blocks):
            x = blk(x, task_id, register_blk==i, get_feat=get_feat, get_cur_feat=get_cur_feat)

        x = self.norm(x)
        
        return x, prompt_loss



def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    # pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
    pretrained_cfg = resolve_pretrained_cfg(variant)
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT_lora_fft, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model



class SiNet(nn.Module):

    def __init__(self, args):
        super(SiNet, self).__init__()

        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, n_tasks=args["total_sessions"], rank=args["rank"])
        self.image_encoder =_create_vision_transformer('vit_base_patch16_224_in21k', pretrained=True, **model_kwargs)
        # print(self.image_encoder)
        # exit()

        self.class_num = 1
        self.class_num = args["init_cls"]
        self.classifier_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

        self.classifier_pool_backup = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

        # self.prompt_pool = CodaPrompt(args["embd_dim"], args["total_sessions"], args["prompt_param"])

        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image, task=None):
        if task == None:
            image_features, _ = self.image_encoder(image, self.numtask-1)
        else:
            image_features, _ = self.image_encoder(image, task)
        image_features = image_features[:,0,:]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image, get_feat=False, get_cur_feat=False, fc_only=False):
        if fc_only:
            fc_outs = []
            for ti in range(self.numtask):
                fc_out = self.classifier_pool[ti](image)
                fc_outs.append(fc_out)
            return torch.cat(fc_outs, dim=1)

        logits = []
        image_features, prompt_loss = self.image_encoder(image, task_id=self.numtask-1, get_feat=get_feat, get_cur_feat=get_cur_feat)
        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)
        for prompts in [self.classifier_pool[self.numtask-1]]:
            logits.append(prompts(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features,
            'prompt_loss': prompt_loss
        }

    def interface(self, image, task_id = None):
        image_features, _ = self.image_encoder(image, task_id=self.numtask-1 if task_id is None else task_id)

        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits,1)
        return logits
    
    def interface1(self, image, task_ids):
        logits = []
        for index in range(len(task_ids)):
            image_features, _ = self.image_encoder(image[index:index+1], task_id=task_ids[index].item())
            image_features = image_features[:,0,:]
            image_features = image_features.view(image_features.size(0),-1)

            logits.append(self.classifier_pool_backup[task_ids[index].item()](image_features))

        logits = torch.cat(logits,0)
        return logits

    def interface2(self, image_features):

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits,1)
        return logits

    def update_fc(self, nb_classes):
        self.numtask +=1

    def classifier_backup(self, task_id):
        self.classifier_pool_backup[task_id].load_state_dict(self.classifier_pool[task_id].state_dict())

    def classifier_recall(self):
        self.classifier_pool.load_state_dict(self.old_state_dict)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
