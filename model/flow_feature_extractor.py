import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from model.flow_guid import MotionFormerBlock
from model.flow.spynet import SPyNet
import time
from .utils import FeatureFlowAttention

def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]

def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid

def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depths=2,act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for i in range(depths):
            if i == 0:
                layers.append(nn.Conv2d(in_dim, out_dim, 3,1,1))
            else:
                layers.append(nn.Conv2d(out_dim, out_dim, 3,1,1))
            layers.extend([
                act_layer(out_dim),
            ])
        self.conv = nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=3, stride=2, in_chans=256, embed_dim=512):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                             kernel_size=patch_size, 
                             stride=stride,
                             padding=(patch_size[0] // 2, patch_size[1] // 2))
        
        self.norm = nn.LayerNorm(embed_dim)
        
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

    def forward(self, x):
      
        x = x.contiguous()
        x = self.proj(x)
        _, _, H, W = x.shape
        
      
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
            
        return x, H, W

class CrossScalePatchEmbed(nn.Module):
    def __init__(self, in_dims=[16,32,64], embed_dim=768):
        super().__init__()
        base_dim = in_dims[0]
        
        layers = []
        for i in range(len(in_dims)):
            for j in range(2 ** i):
                layers.append(nn.Conv2d(in_dims[-1-i], base_dim, 3, 2**(i+1), 1+j, 1+j))
        self.layers = nn.ModuleList(layers)
        self.proj = nn.Conv2d(base_dim * len(layers), embed_dim, 1, 1)
        self.norm = nn.LayerNorm(embed_dim)

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

    def forward(self, xs):
        ys = []
        k = 0
        for i in range(len(xs)):
            for _ in range(2 ** i):
          
                current_feat = xs[-1-i].contiguous()
                ys.append(self.layers[k](current_feat))
                k += 1
       
        ys = [y.contiguous() for y in ys]
        x = self.proj(torch.cat(ys, 1).contiguous())
        _, _, H, W = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
        
        return x, H, W




class BiFlow(nn.Module):
    def __init__(self,local_corr_radius=3,feature_channels=128,
                 local_window_attn=False,local_window_radius=1,match_flow=False ):
        super().__init__()

        self.feature_channels = feature_channels
        self.local_window_attn=local_window_attn
        self.local_window_radius=local_window_radius
        self.local_radius = local_corr_radius
        self.match_flow = match_flow

   
        self.flow_estimator = nn.Sequential(
            nn.Conv2d(feature_channels * 2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1)  # 输出2通道光流
        )




    def forward(self, x, pred_bidir_flow=False):
       
        B = x.shape[0]
        feature0 = x

        feature1 = torch.cat([x[B // 2:], x[:B // 2]])


        init_flows = {}
        init_flows['flow_ds8_'], init_flows['flow_ds16_'] = [], []

        if self.flow_estimator is None:
            # correlation and softmax
            if self.local_radius == -1:  # global matching
                flow = self.global_correlation_softmax(feature0, feature1, pred_bidir_flow)
            else:  # local matching
                flow = self.local_correlation_softmax(feature0, feature1, self.local_radius)   #[B, 2, H, W]

            #feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation
            flow_finall = self.feature_flow_attn(feature0, flow.detach(), local_window_attn=self.local_window_attn, local_window_radius=self.local_window_radius)
        else:
            flow_finall = self.flow_estimator(torch.cat([feature0, feature1], 1))



        flow_ds8_ = flow_finall
        #flow_ds8_ = F.interpolate(flow, scale_factor=1/8, mode='bilinear', align_corners=False).contiguous() * 1/8
        flow_ds16_ = F.interpolate(flow_ds8_, scale_factor=1/2, mode='bilinear', align_corners=False).contiguous() * 1/2
        


        init_flows['flow_ds8_'] = flow_ds8_
        init_flows['flow_ds16_'] = flow_ds16_
        
        return init_flows





class MotionFormer(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=[0, 0, 0, 64, 128], num_heads=[8, 16],  window_size=2, alpha=0.5, 
                  scale=[2, 2], local_corr_radius=[3,3], flow_align=True, offset_align=False, linear_flow=True,
                depths=[2, 2, 2, 4, 4], matching_type='global_corr', downsample_mode='bilinear', qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, spynet_pretrained=None,  **kwarg):
        super().__init__()

        self.depths = depths
        self.num_stages = len(embed_dims)


       
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.flow_cal_downsample = downsample_mode
        
        #self.spynet = SPyNet(pretrained=spynet_pretrained)  # flow estimation network
        
        self.cal_flow = BiFlow(local_corr_radius=3,
                 feature_channels=embed_dims[3],
                 local_window_attn=False,
                 local_window_radius=1,
                 match_flow=False)

                 
       
        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = CrossScalePatchEmbed(embed_dims[:i],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                
                    

                    block = nn.ModuleList([MotionFormerBlock(
                        dim=embed_dims[i], motion_dim = motion_dims[i], num_heads=num_heads[i-self.conv_stages], window_size= window_size, 
                        alpha=alpha, 
                        scale=scale[i-self.conv_stages], local_corr_radius = local_corr_radius[i-self.conv_stages], 
                        flow_align = flow_align, offset_align = offset_align, linear_flow = linear_flow, matching_type = matching_type,
                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer)
                        for j in range(depths[i])])



                    norm = norm_layer(embed_dims[i])
                    setattr(self, f"norm{i + 1}", norm)
                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

       # self.register_buffer('coord_grid_cache', None)

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


    def forward(self, x1, x2):
      

        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        motion_features = []
        motion_matching_features = []
        appearence_features = []
        xs = []
        
        
        #init_flows = {}
        #init_flows = self.compute_init_flow(x1, x2, init_flows)
     

        for i in range(self.num_stages):
            motion_features.append([])
            motion_matching_features.append([])
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            norm = getattr(self, f"norm{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
                xs.append(x)
            else:
                if i == self.conv_stages:
                    x, H, W = patch_embed(xs)
                    init_flows = self.cal_flow(x, pred_bidir_flow=False)
                else:
                    x, H, W = patch_embed(x)
           
                current_flow = None
                if i == 3:  # stage 4
                    current_flow = init_flows['flow_ds8_'][0] if isinstance(init_flows['flow_ds8_'], list) else init_flows['flow_ds8_']
                    if current_flow.device != x.device:
                        current_flow = current_flow.to(x.device)
                    current_flow = current_flow.contiguous()
                elif i == 4:  # stage 5
                    current_flow = init_flows['flow_ds16_'][0] if isinstance(init_flows['flow_ds16_'], list) else init_flows['flow_ds16_']
                    if current_flow.device != x.device:
                        current_flow = current_flow.to(x.device)
                    current_flow = current_flow.contiguous()
    
                for blk in block:
                    assert x.is_contiguous(), f" x before block, shape: {x.shape}"
                    if current_flow is not None:
                        assert current_flow.is_contiguous(), f"current_flow  before block, shape: {current_flow.shape}"

                    
                    x, x_motion_matching = blk(x, H, W, B, current_flow)   #(B, N, C)
                    

                    #motion_features[i].append(x_motion)
                    motion_matching_features[i].append(x_motion_matching)
                    
                x = norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
                
                #motion_features[i] = torch.cat(motion_features[i], 1)
               
                motion_matching_features[i] = torch.cat(motion_matching_features[i], 1)
                
            appearence_features.append(x)
            

        return appearence_features, motion_matching_features




def feature_extractor(**kargs):
    model = MotionFormer(**kargs)
    return model