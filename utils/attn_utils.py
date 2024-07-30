# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..  
# *************************************************************************

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from diffusers.utils.import_utils import is_xformers_available
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None
class AttentionBase:

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, is_cross, place_in_unet, num_heads, batch, **kwargs):
        out = self.forward(q, k, v, is_cross, place_in_unet, num_heads, batch, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, is_cross, place_in_unet, num_heads, batch, **kwargs):
        if hasattr(self,"flag") and not is_cross:
            q, k, v = map(lambda t: rearrange(t, 'b n D -> (b n) D'), (q, k, v))
            q, k, v = map(lambda t: rearrange(t, 'l n (h d) ->l n h d', h=num_heads), (q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)))
            out = self._memory_efficient_attention_xformers(q, k, v,attention_mask=None,num_heads=num_heads,batch=batch)[0]
            out = rearrange(out, '(b n) h d -> b n h d',b=batch)
        else:
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=num_heads), (q, k, v))
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
            # out[mid:mid+1] = xformers.ops.memory_efficient_attention(q[mid+1:mid+2], k[mid+1:mid+2], v[mid+1:mid+2], attn_bias=None)
            # torch.cat([k[15:16],k[16:17]], dim=1) torch.cat([v[15:16],v[16:17]]
            # out[16:17] = xformers.ops.memory_efficient_attention(q[16:17], torch.cat([k[16:17],k[15:16]], dim=1), torch.cat([v[16:17],v[15:16]], dim=1), attn_bias=None)
        # out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        # out = rearrange(out, '(b h) n d -> b n (h d)',h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

class MutualSelfAttentionControl(AttentionBase):

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, guidance_scale=7.5):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
        """
        super().__init__()
        self.total_steps = total_steps
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, 16))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        # store the guidance scale to decide whether there are unconditional branch
        self.guidance_scale = guidance_scale
        print("step_idx: ", self.step_idx)
        print("layer_idx: ", self.layer_idx)

    def forward(self, q, k, v, is_cross, place_in_unet, num_heads, batch, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, is_cross, place_in_unet, num_heads, batch, **kwargs)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=num_heads), (q, k, v))
        if self.guidance_scale > 1.0:
            qu, qc = q[0:2], q[2:4]
            ku, kc = k[0:2], k[2:4]
            vu, vc = v[0:2], v[2:4]

            # merge queries of source and target branch into one so we can use torch API
            qu = torch.cat([qu[0:1], qu[1:2]], dim=2)
            qc = torch.cat([qc[0:1], qc[1:2]], dim=2)

            out_u = F.scaled_dot_product_attention(qu, ku[0:1], vu[0:1], attn_mask=None, dropout_p=0.0, is_causal=False)
            out_u = torch.cat(out_u.chunk(2, dim=2), dim=0) # split the queries into source and target batch
            out_u = rearrange(out_u, 'b h n d -> b n (h d)')

            out_c = F.scaled_dot_product_attention(qc, kc[0:1], vc[0:1], attn_mask=None, dropout_p=0.0, is_causal=False)
            out_c = torch.cat(out_c.chunk(2, dim=2), dim=0) # split the queries into source and target batch
            out_c = rearrange(out_c, 'b h n d -> b n (h d)')

            out = torch.cat([out_u, out_c], dim=0)
        else:
            # k_contrl = torch.cat([k[:1], k[-1:]], dim=2)
            # q_contrl = torch.cat([q[:1], q[-1:]], dim=2)
            # out = F.scaled_dot_product_attention(q, k_contrl, q_contrl, attn_mask=None, dropout_p=0.0, is_causal=False)
            # out = rearrange(out, 'b h n d -> b n (h d)')

            out = torch.zeros_like(q)
            out[:1] = xformers.ops.memory_efficient_attention(q[:1], k[:1], v[:1], attn_bias=None)
            out[-1:] = xformers.ops.memory_efficient_attention(q[-1:], k[-1:], v[-1:], attn_bias=None)
            k_time = torch.cat([k[:1], k[-1:]], dim=1)
            v_time = torch.cat([v[:1], v[-1:]], dim=1)
            q_time = rearrange(q[1:-1], 'b n h d -> (b n) h d').unsqueeze(0)
            out_time = xformers.ops.memory_efficient_attention(q_time, k_time, v_time, attn_bias=None)
            out[1:-1] = rearrange(out_time[0], '(b n) h d -> b n h d',b=batch-2)
            # if self.cur_step>25:
            #     mid = self.mid
                # out[mid-2:mid-1] = xformers.ops.memory_efficient_attention(q[mid-2:mid-1], torch.cat([k_time,k[mid+1:mid+2]], dim=1), torch.cat([v_time,v[mid+1:mid+2]], dim=1), attn_bias=None)
                # out[mid-1:mid] = xformers.ops.memory_efficient_attention(q[mid-1:mid], torch.cat([k_time,k[mid:mid+1]], dim=1), torch.cat([v_time,v[mid:mid+1]], dim=1), attn_bias=None)
                # out[mid:mid+1] = xformers.ops.memory_efficient_attention(q[mid:mid+1], torch.cat([k_time,k[mid-2:mid-1]], dim=1), torch.cat([v_time,v[mid-2:mid-1]], dim=1), attn_bias=None) 
                # out[mid+1:mid+2] = xformers.ops.memory_efficient_attention(q[mid+1:mid+2], torch.cat([k_time,k[mid-2:mid-1]], dim=1), torch.cat([v_time,v[mid-2:mid-1]], dim=1), attn_bias=None)  
            
            # q = torch.cat([q[0:1], q[1:2]], dim=2)
            # out = F.scaled_dot_product_attention(q, k[0:1], v[0:1], attn_mask=None, dropout_p=0.0, is_causal=False)
            # out = torch.cat(out.chunk(2, dim=2), dim=0) # split the queries into source and target batch
            # out = rearrange(out, 'b h n d -> b n (h d)')
        return out

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask, num_heads, batch):
        # TODO attention_mask
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        return hidden_states
    
    
# forward function for default attention processor
# modified from __call__ function of AttnProcessor in diffusers
def override_attn_proc_forward(attn, editor, place_in_unet):
    def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
        """
        The attention is similar to the original implementation of LDM CrossAttention class
        except adding some modifications on the attention
        """
        if encoder_hidden_states is not None:
            context = encoder_hidden_states
        if attention_mask is not None:
            mask = attention_mask

        to_out = attn.to_out
        if isinstance(to_out, nn.modules.container.ModuleList):
            to_out = attn.to_out[0]
        else:
            to_out = attn.to_out

        h = attn.heads
        q = attn.to_q(x)
        is_cross = context is not None
        context = context if is_cross else x
        k = attn.to_k(context)
        v = attn.to_v(context)

        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # the only difference
        out = editor(
            q, k, v, is_cross, place_in_unet,
            attn.heads, batch=34,scale=attn.scale)
        out = rearrange(out, 'b n h d -> b n (h d)',h=attn.heads)
        return to_out(out)

    return forward

def normalize_for_grid_sample(coords, h, w):
    device = coords.device
    coords_normed = coords / torch.tensor([w-1., h-1.]).to(device) * 2. - 1.
    return coords_normed

# forward function for lora attention processor
# modified from __call__ function of LoRAAttnProcessor2_0 in diffusers v0.17.1
def override_lora_attn_proc_forward(attn, editor, place_in_unet):
 
    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, lora_scale=0.8):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        is_cross = encoder_hidden_states is not None

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) + lora_scale * attn.processor.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + lora_scale * attn.processor.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + lora_scale * attn.processor.to_v_lora(encoder_hidden_states)

        # the only difference
        hidden_states = editor(
            query, key, value, is_cross, place_in_unet,
            attn.heads, batch=batch_size, scale=attn.scale)

        hidden_states = rearrange(hidden_states, 'b n h d -> b n (h d)',h=attn.heads)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + lora_scale * attn.processor.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    return forward

def register_attention_editor_diffusers(model, editor: AttentionBase, attn_processor='attn_proc'):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                if attn_processor == 'attn_proc':
                    net.forward = override_attn_proc_forward(net, editor, place_in_unet)
                elif attn_processor == 'lora_attn_proc':
                    net.forward = override_lora_attn_proc_forward(net, editor, place_in_unet)
                else:
                    raise NotImplementedError("not implemented")
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count