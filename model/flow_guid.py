import torch.nn as nn
import torch
import torch.nn.functional as F


import math
import warnings

from timm.models.layers import DropPath, trunc_normal_ 
from model.geometry import coords_grid, generate_window_grid, normalize_coords



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], C)
    )
    return windows



def window_reverse(windows, window_size, H, W):
    nwB, N, C = windows.shape
    windows = windows.view(-1, window_size[0], window_size[1], C)
    B = int(nwB / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


#pad
def pad_if_needed(x, size, window_size):
    n, h, w, c = size
    pad_h = math.ceil(h / window_size[0]) * window_size[0] - h
    pad_w = math.ceil(w / window_size[1]) * window_size[1] - w
    if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
        img_mask = torch.zeros((1, h + pad_h, w + pad_w, 1))  # 1 H W 1
        h_slices = (
            slice(0, pad_h // 2),
            slice(pad_h // 2, h + pad_h // 2),
            slice(h + pad_h // 2, None),
        )
        w_slices = (
            slice(0, pad_w // 2),
            slice(pad_w // 2, w + pad_w // 2),
            slice(w + pad_w // 2, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, window_size
        )  # nW, window_size*window_size, 1
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)
        ).masked_fill(attn_mask == 0, float(0.0))
        return nn.functional.pad(
            x,
            (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        ), attn_mask
    return x, None


#depad
def depad_if_needed(x, size, window_size):
    n, h, w, c = size
    pad_h = math.ceil(h / window_size[0]) * window_size[0] - h
    pad_w = math.ceil(w / window_size[1]) * window_size[1] - w
    if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
        return x[:, pad_h // 2: pad_h // 2 + h, pad_w // 2: pad_w // 2 + w, :].contiguous()
    return x



class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()

        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
       
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.reshape(B, C, -1).transpose(1, 2)

        return x


class Mlp(nn.Module):
 
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)        # Dropout
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)      


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
      
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def normal_init(module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)


class OffsetAlignment(nn.Module):


    def __init__(self, in_channels,
                 kernel_size=3, 
                 local_window=3,
                 sim_type='cos', 
                 sample_type='sample',
                 ):
        super().__init__()

        self.local_window = local_window
        self.sim_type = sim_type  # similarity type
        self.sample_type = sample_type

       
        out_channels = 2

        if self.direction_feat == 'sim':
            self.offset = nn.Conv2d(local_window ** 2 - 1, out_channels, kernel_size=kernel_size,
                                    padding=kernel_size // 2)
   
            raise NotImplementedError
        normal_init(self.offset, std=0.001)

        if self.direction_feat == 'sim':
            self.direct_scale = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        
        else:
            raise NotImplementedError
        constant_init(self.direct_scale, val=0.)

     
        self.norm = nn.GroupNorm(in_channels // 8, in_channels)
        




    def _get_init_pos(self, device):
      
            h = torch.arange(-1.0, 1.0 + 1e-4, 1.0, device=device)
            grid = torch.stack(torch.meshgrid([h, h])).transpose(1, 2)
            init_pos = grid[:, 1:2, 1:2].reshape(1, 2, 1, 1)
            return init_pos.contiguous()

    

    def forward(self, aligned_feat, original_feat, flow):
  
    

        aligned_feat = self.norm(aligned_feat)

        # compute similarity
        if self.direction_feat == 'sim':
            sim = self.compute_similarity(aligned_feat, self.local_window, dilation=2, sim='cos')

        init_pos = self._get_init_pos(aligned_feat.device)
        offset = (self.offset(sim)) * (self.direct_scale(sim)).sigmoid() + init_pos

        # sample
        if self.sample_type == 'sample':
            out = self.sample(aligned_feat, offset)

        return out

    def sample(self, x, offset):

        assert x.shape[0] == offset.shape[0], "Batch sizes don't match"
        assert offset.shape[1] == 2, "Offset should have 2 channels"
        assert x.shape[2:] == offset.shape[2:], "Spatial dimensions don't match"
        
        B, _, H, W = offset.shape

   
        coords_h = torch.arange(H) + 0.5  # [H]
        coords_w = torch.arange(W) + 0.5  # [W]
  
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2)  # [2,H,W]
    
        coords = coords.unsqueeze(0)  # [1,2,H,W]
        coords = coords.type(x.dtype).to(x.device)

     
        coords = coords + offset  # [B,2,H,W]
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device
                           ).view(1, 2, 1, 1)  # [1,2,1,1]
        coords = 2 * coords / normalizer - 1  # [B,2,H,W]
        coords = coords.permute(0, 2, 3, 1)  # [B,H,W,2]
        sampled_feat = F.grid_sample(x, coords, mode='nearest', align_corners=False, padding_mode="border")  # [B,C,H,W]

        return sampled_feat

    def double_sample(self, x, flow, offset):

        B, _, H, W = x.shape

    
        coords_h = torch.arange(H, device=x.device) + 0.5  # [0.5, 1.5, ..., H-0.5]
        coords_w = torch.arange(W, device=x.device) + 0.5  # [0.5, 1.5, ..., W-0.5]
        grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0)  # [2,H,W]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B,2,H,W]

        vgrid = grid + flow + offset  

        vgrid_x = 2.0 * vgrid[:,:1] / max(W - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:,1:] / max(H - 1, 1) - 1.0
        vgrid_scaled = torch.cat((vgrid_x, vgrid_y), dim=1)  # [B,2,H,W]
        vgrid_scaled = vgrid_scaled.permute(0, 2, 3, 1)  # [B,H,W,2]

        sampled_feat = F.grid_sample(
            x, 
            vgrid_scaled, 
            mode='nearest', 
            align_corners=False, 
            padding_mode="border"
        )
        return sampled_feat

    def compute_similarity(self,input_tensor, k=3, dilation=1, sim='cos'):
  
        B, C, H, W = input_tensor.shape
        unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation)  # B, CxKxK, HW
        unfold_tensor = unfold_tensor.reshape(B, C, k ** 2, H, W)

        if sim == 'cos':
          similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
        else:
            raise NotImplementedError

        similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)
        similarity = similarity.view(B, k * k - 1, H, W)
        return similarity


# DAM
class DualAlignment(nn.Module):
    def __init__(self, dim, flow_align = True, offset_align = False):
        super().__init__()
        self.dim = dim

        self.flow_align = flow_align
        self.offset_align = offset_align

        self.offset_aligned = OffsetAlignment(in_channels = dim,
                                                          kernel_size=3, 
                                                          local_window=3,
                                                          sim_type='cos', 
                                                          sample_type='sample')

    def forward(self, x0, x1, flow):
        
        _, _, H, W = x0.shape
        
        #Flow Alignment
        if self.flow_align:
    
            grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
            grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
            grid.requires_grad = False
            grid = grid.type_as(x1)       # torch.Size([32, 32, 2])

            vgrid = grid + flow.permute(0, 2, 3, 1)
            vgrid_x = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            out = F.grid_sample(x1, vgrid_scaled, mode='nearest', align_corners=False, padding_mode="border")  

        if self.offset_align:
            # offset_aligned   flow_aligned_feat [2B,C,H,W]
            out = self.offset_aligned(out, x1, flow)  # [2B,C,H,W]
            

        return out




class RSAMAttn(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads)
        self.dim = dim
        self.G_heads = int(num_heads * alpha)
        self.G_dim = self.G_heads * head_dim
        self.L_heads = num_heads - self.G_heads
        self.L_dim = self.L_heads * head_dim
        self.ws = window_size

        if self.ws == 1:
           
            self.L_heads = 0
            self.L_dim = 0
            self.G_heads = num_heads
            self.G_dim = dim

        self.scale = qk_scale or head_dim ** -0.5


        if self.G_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.G_q = nn.Linear(self.dim, self.G_dim, bias=qkv_bias)
            self.G_kv = nn.Linear(self.dim, self.G_dim * 2, bias=qkv_bias)
            self.G_proj = nn.Linear(self.G_dim, self.G_dim)

    
        if self.L_heads > 0:
            self.L_q = nn.Linear(self.dim, self.L_dim, bias=qkv_bias)
            self.L_kv = nn.Linear(self.dim, self.L_dim * 2, bias=qkv_bias)
            self.L_proj = nn.Linear(self.L_dim, self.L_dim)


    def LAtt(self, x, x_reverse):
        B, H, W, C = x.shape
   
        
        h_group, w_group = H // self.ws, W // self.ws    #ws=2
        total_groups = h_group * w_group  #16*16
 
   
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)    
        x_reverse = x_reverse.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)   

        q = self.L_q(x).reshape(B, total_groups, -1, self.L_heads, self.L_dim // self.L_heads).permute(0, 1, 2, 3, 4)
        kv = self.L_kv(x_reverse).reshape(B, total_groups, -1, 2, self.L_heads, self.L_dim // self.L_heads).permute(3, 0, 1, 2, 4, 5)

    
        k, v = kv[0], kv[1] 
    
        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.L_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.L_dim)
        x = self.L_proj(x)
        return x

    def GAtt(self, x, x_reverse):
        B, H, W, C = x.shape

        q = self.G_q(x).reshape(B, H * W, self.G_heads, self.l_dim // self.G_heads).permute(0, 2, 1, 3)

        if self.ws > 1:
            x_reverse_ = x_reverse.permute(0, 3, 1, 2)
            x_reverse_ = self.sr(x_reverse_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.G_kv(x_reverse_).reshape(B, -1, 2, self.G_heads, self.l_dim // self.G_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.G_kv(x_reverse).reshape(B, -1, 2, self.G_heads, self.G_dim // self.G_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.G_dim)
        x = self.G_proj(x)
        return x

    def forward(self, x, x_reverse, H, W):
        B, N, C = x.shape

        x = x.reshape(B, H, W, C)
        x_reverse = x_reverse.reshape(B, H, W, C)
        # pad feature maps to multiples of window size
        pad_G = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_G, pad_r, pad_t, pad_b))
        x_reverse = F.pad(x_reverse, (0, 0, pad_G, pad_r, pad_t, pad_b))
      

        
        GAtt_out = self.GAtt(x, x_reverse)
        LAtt_out = self.LAtt(x, x_reverse)
       
        if pad_r > 0 or pad_b > 0:
            x = torch.cat((GAtt_out[:, :H, :W, :], LAtt_out[:, :H, :W, :]), dim=-1)
        else:
            x = torch.cat((GAtt_out, LAtt_out), dim=-1)

        if not hasattr(self, '_flops_printed'):
            self.flops(N)
            self._flops_printed = True
        
        x = x.reshape(B, N, C)
        return x

  


class AttnBlock(nn.Module):

    def __init__(self, dim, num_heads, window_size=7, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, alpha=0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = RSAMAttn(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, proj_drop=drop, window_size=window_size, alpha=alpha)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DWMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, x_reverse, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x_reverse), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x    #(B, N, C)





class MotionFormerBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 motion_dim, 
                 num_heads, 
                 window_size,
                 alpha,
                 scale=2, 
                 local_corr_radius = 3, 
                 flow_align = True, 
                 offset_align = False, 
                 linear_flow = True, 
                 matching_type = 'global_corr',
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                            ):     
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dual_align = DualAlignment(dim = dim, flow_align = flow_align, offset_align = offset_align)

        self.attn = AttnBlock(dim=dim, num_heads=num_heads,  window_size=window_size, 
                 mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path= drop_path,
                 act_layer=nn.GELU, norm_layer=norm_layer, alpha=alpha)
        

        self.scale = scale       
        self.motion_proj = nn.Linear(motion_dim, motion_dim)
        self.grid_proj = nn.Linear(2, motion_dim, bias=qkv_bias)
        self.linear_flow = linear_flow                     
        self.local_radius = local_corr_radius
        self.matching = matching_type

     #Motion Compensation Module
    def global_correlation_softmax(self, feature0, feature1, pred_bidir_flow=False):
      
        b, c, h, w = feature0.shape
        
     
        feature0 = feature0.view(b, c, -1).permute(0, 2, 1).contiguous()  
        feature1 = feature1.view(b, c, -1).contiguous() 


        correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w).contiguous() / (c ** 0.5)


        init_grid = coords_grid(b, h, w).to(correlation.device)  
        init_grid_embed = self.grid_proj(init_grid.permute(0, 2, 3, 1).contiguous()) 
        init_grid_embed = init_grid_embed.permute(0, 3, 1, 2).contiguous()  

        grid = init_grid.view(b, 2, -1).permute(0, 2, 1).contiguous()  
        grid_embed = init_grid_embed.permute(0, 2, 3, 1).reshape(b, h * w, -1).contiguous() 

        correlation = correlation.view(b, h * w, h * w).contiguous() 

        if pred_bidir_flow:
            correlation = torch.cat((correlation, correlation.permute(0, 2, 1).contiguous()), dim=0)
            init_grid = init_grid.repeat(2, 1, 1, 1)
            grid = grid.repeat(2, 1, 1)
            b = b * 2

        prob = F.softmax(correlation, dim=-1)  

  
        motion_feat = torch.matmul(prob, grid_embed) 
        motion_feat = motion_feat.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  

        motion = self.motion_proj((motion_feat - init_grid_embed).permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        return motion


   # def local_correlation_softmax(self, feature0, feature1, local_radius, padding_mode='zeros'):
        #b, c, h, w = feature0.size()

       # return motion
    
    def forward(self, x, H, W, B, init_flows):
    
        x_reverse = torch.cat([x[B // 2:], x[:B // 2]])
        x_reverse_aligned = self.dual_align(x, x_reverse, init_flows)
        x_reverse_aligned = x_reverse_aligned.flatten(2).transpose(1, 2) 

        attn_out = self.attn(x, x_reverse_aligned, H, W)  
    
        B_attn = attn_out.shape[0]
        attn_out = attn_out.reshape(B_attn, H, W, -1).permute(0, 3, 1, 2).contiguous()  
        
        if self.matching == 'global_corr':
            if self.scale == 2:
                h, w = H // self.scale, W // self.scale
                x_matching = F.interpolate(attn_out, size=(h, w), mode='bilinear', align_corners=False)
                x_matching_reverse = torch.cat([x_matching[B // 2:], x_matching[:B // 2]])
                motion_matching = self.global_correlation_softmax(x_matching, x_matching_reverse)
                motion_matching = F.interpolate(motion_matching, size=(H, W), mode='bilinear', align_corners=False)
            else:
                motion_matching = self.global_correlation_softmax(attn_out, 
                    torch.cat([attn_out[B // 2:], attn_out[:B // 2]]))
        else:  # local_corr
            x_in = torch.cat([attn_out[B // 2:], attn_out[:B // 2]])
            motion_matching = self.local_correlation_softmax(attn_out, x_in, local_radius=self.local_radius)

      
        return attn_out, motion_matching



class dwConv(nn.Module):
    def __init__(self, dim=768):
        super(dwConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):

        x = self.dwconv(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class DWMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



