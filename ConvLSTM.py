import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath, to_2tuple
from einops import rearrange


__all__ = [
    'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
    'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
]
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

    
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size) 
        self.img_size = img_size 
        self.patch_size = patch_size 
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1) 
        self.num_patches = (self.grid_size[0]//2 * self.grid_size[1]//2) 
        self.flatten = flatten 

        self.proj1 = nn.Sequential(
            nn.Conv2d(10, embed_dim, kernel_size=4, stride=4),  
            nn.ReLU(inplace=True), 
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim*2, kernel_size=4, stride=4), 
            nn.ReLU(inplace=True), 
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim*2, kernel_size=2, stride=2), 
        )

        self.norm = norm_layer(embed_dim*2) if norm_layer else nn.Identity() 

    def forward(self, x):

        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x1 = self.proj1(x) 
        x2 = self.proj2(x1) 
        x3 = self.proj3(x2) 

        return x3

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2 
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,  
            out_channels=4 * self.hidden_dim,              
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)   
        f = torch.sigmoid(cc_f)    
        o = torch.sigmoid(cc_o)    
        g = torch.tanh(cc_g)      

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
    
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        b, seq_len, _,h, w = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx] 
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
            
        return layer_output_list[0][:,-1,...]

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not isinstance(kernel_size, tuple):
            raise ValueError('`kernel_size` must be tuple.')

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class ConvLSTMModel(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 embed_dim=256, 
                 channels=10, 
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs) 
        super().__init__()

        self.d_model = self.num_features = self.embed_dim = embed_dim 

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=16, stride=16, in_chans=channels, embed_dim=embed_dim)


        self.convlstm = ConvLSTM(input_dim=384, hidden_dim=embed_dim*2, kernel_size=(3, 3), num_layers=2, batch_first=True)
        self.proj=nn.Linear(self.embed_dim*2, self.embed_dim*2)


        self.importance_proj1 = nn.Sequential(
            nn.Linear(self.embed_dim*2, 1),
            nn.Sigmoid() 
        )
        self.importance_proj2 = nn.Sequential(
            nn.Linear(self.embed_dim*2, 1),
            nn.Sigmoid()  
        )
       
        self.x_head = nn.Sequential(
            nn.Linear(self.embed_dim*2, self.embed_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1, bias=True)
        )

        self.patch_embed.apply(segm_init_weights)

        
    def forward(self, x):
        
        b, t, g, _, _, _ = x.shape 
        x = rearrange(x, 'b t g c h w -> (b t g) c h w')
        x = self.patch_embed(x) 
        x = rearrange(x, '(b t g) c h w -> (b g) t c h w',g=g,t=t)
        x = self.convlstm(x) 
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x) 
        att1=self.importance_proj1(x)
        x = (x * att1).sum(dim=1) 
        x = x.unsqueeze(0)

        att2=self.importance_proj2(x)
        x = (x * att2).sum(dim=1)

        x = self.x_head(x)

        return x

