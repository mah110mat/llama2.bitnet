# https://github.com/Hajime-Y/BitNet-b158/blob/main/mybitnet/bitnet/bitnet.py
import torch
from torch import nn
from typing import Tuple
import numpy as np
import sys

class BitRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BitRMSNorm is equivalent to LlamaRMSNorm and T5LayerNorm
        refers: https://github.com/huggingface/transformers/blob/c5f0288bc7d76f65996586f79f69fba8867a0e67/src/transformers/models/llama/modeling_llama.py#L76C1-L90C59
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, rms_norm_eps=1e-6, bits=8, flg_before_linear=True):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.layernorm = BitRMSNorm(hidden_size=in_features, eps=rms_norm_eps)
        self.bits = bits
        self.Qb = 2 ** (self.bits - 1)
        self.rQb = (1 / self.Qb)
        self.flg_before_linear = flg_before_linear
        self.epsilon = 1e-6  # overflow防止のための小さな値

    def absmax_quantize(self, x: torch.Tensor, Qb: int, epsilon: float) -> Tuple[torch.Tensor, float]:
        if self.flg_before_linear:
            # パターン①：　通常は[-Qb, Qb]にスケール: 式(4), (5)を適用
            gamma = torch.abs(x).max().clamp(min=epsilon)
            x_scaled = x * Qb / gamma
            x_q = torch.round(x_scaled).clamp(-Qb, Qb - 1)
        else:
            # パターン②：　Reluなどの非線形関数前の場合は[0, Qb]にスケール：　式(6)を適用
            # 論文中には記載はないですが、スケールが異なるためスケーリングの基準として使っているgammaもetaを反映した値にすべきだと考えます。
            eta = x.min()
            gamma = torch.abs(x - eta).max().clamp(min=epsilon)
            x_scaled = (x - eta) * Qb / gamma
            x_q = torch.round(x_scaled).clamp(0, Qb - 1)
        # STE
        x_q = (x_q - x_scaled).detach() + x_scaled
        return x_q, gamma
        
    # 独自のsign関数の定義
    # torch.signは0を0として扱ってしまう。custom_signはW>0を+1に、W≦0を-1とする。
    def custom_sign(self, x):
        return (x > 0).to(torch.int8) * 2 - 1

    def quantize_weights(self, weight: torch.Tensor, epsilon: float) -> Tuple[torch.Tensor, float]:
        # 式(3): alphaの計算
        alpha = weight.mean()

        # 式(1),(2): 重みの中心化とバイナリ化
        weight_centered = weight - alpha
        weight_binarized = self.custom_sign(weight_centered)

        # 式(12): betaの計算
        beta = weight.abs().mean()

        # STE (weight_binarizedとスケールを合わせるためweight_centeredをweight_scaledにスケールしています。)
        weight_scaled = weight_centered / (weight_centered.abs().max().clamp(min=epsilon))
        weight_binarized = (weight_binarized - weight_scaled).detach() + weight_scaled

        return weight_binarized, beta
        
    def forward(self, x):
        # 1. LayerNorm (input: x, output: x_norm)
        #import pdb; pdb.set_trace()
        x_norm = self.layernorm(x)

        # 2. Absmax Quatization (input: x_norm, output: x_q, gamma)
        x_q, gamma = self.absmax_quantize(x_norm, self.Qb, self.epsilon)

        # 3. 1-bit Weights化 (input: -, output: w_q, beta)
        w_q, beta = self.quantize_weights(self.weight, self.epsilon)

        ## 4. テンソル積(⊗) (input: x_q,w_q, output: x_matmul)
        ## 5. Dequantization (input: x_matmul,beta,gamma, output: output)
        #x_matmul = torch.nn.functional.linear(x_q, w_q, self.bias)
        #output = x_matmul * (beta * gamma * self.rQb)

        x_matmul = torch.nn.functional.linear(x_norm, w_q, self.bias)
        output = x_matmul * beta


        #output = x_matmul * (beta * gamma / self.Qb)
        if np.isnan(self.Qb):
            print("self.Qb: ", self.Qb, beta, gamma, x_matmul)
            sys.exit()
        '''
        #if np.isnan(beta.cpu().numpy()):
        if np.isnan(beta.detach().cpu().numpy()):
            print("beta: ", self.Qb, beta, gamma, x_matmul)
            sys.exit()
        #if np.isnan(gamma.cpu().numpy()):
        if np.isnan(gamma.detach().cpu().numpy()):
            print("gamma: ", self.Qb, beta, gamma, x_matmul)
            sys.exit()
        #if np.sum(np.isnan(x_matmul.cpu().numpy()))!=0:
        if np.sum(np.isnan(x_matmul.detach().cpu().numpy()))!=0:
            print("x_matmul: ", self.Qb, beta, gamma, x_matmul)
            sys.exit()
        #if np.sum(np.isnan(output.cpu().numpy()))!=0:
        if np.sum(np.isnan(output.detach().cpu().numpy()))!=0:
            print("output: ", self.Qb, beta, gamma, x_matmul, output)
            sys.exit()
        '''
        
        return output

class BitLinear158b(BitLinear):
    def __init__(self, in_features, out_features, bias=True, rms_norm_eps=1e-6, bits=8):
        super().__init__(in_features, out_features, bias, rms_norm_eps, bits)
        # 2. BitLinear b158では、[0, Qb]のスケーリングは行わないため、flg_before_linearは使用しません。
        del self.flg_before_linear
        
    # 1. quantize_weightsを{-1, 1}の2値化から{-1, 0, 1}の3値化に修正
    def quantize_weights(self, weight: torch.Tensor, epsilon: float) -> Tuple[torch.Tensor, float]:
        # 式(3): betaの計算
        beta = weight.abs().mean().clamp(min=epsilon)

        # 式(1),(2): 重みの量子化(-1, 0, 1)とクリップ
        # 各値は{-1, 0, +1}の中で最も近い整数に丸められます。
        weight_trinarized = weight / beta
        weight_trinarized = torch.round(weight_trinarized)
        weight_trinarized = torch.clamp(weight_trinarized, -1, 1)

        # STE
        weight_trinarized = (weight_trinarized - weight).detach() + weight

        return weight_trinarized, beta
    
    # 2. BitLinear b158では、[0, Qb]のスケーリングは行わないません。
    def absmax_quantize(self, x: torch.Tensor, Qb: int, epsilon: float) -> Tuple[torch.Tensor, float]:
        # スケールgammaの計算（absmax quantization）
        gamma = torch.abs(x).max().clamp(min=epsilon)

        # 重みの量子化とクリップ
        x_scaled = x * Qb / gamma
        x_q = torch.round(x_scaled)
        x_q = torch.clamp(x_q, -Qb, Qb - 1)
        
        # STE
        x_q = (x_q - x_scaled).detach() + x_scaled
        
        return x_q, gamma
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
