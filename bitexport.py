import struct
import numpy as np
import torch
from typing import Tuple

from export import serialize_fp32
def serialize_b158(file, tensor, msg):

    def quantize_weights(weight: torch.Tensor, epsilon: float) -> Tuple[torch.Tensor, float]:
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

    wt, beta = quantize_weights(tensor, epsilon=1e-6)
    d = wt.detach().cpu().view(-1).to(torch.float32).numpy()
    wt_cpu = wt.detach().cpu()
    wt_np = wt_cpu.numpy()

    b1 = struct.pack('f', beta)
    b0 = struct.pack(f'{len(d)}f', *d)
    #print(f'{msg}, {len(d)}, {beta}')
    file.write(b1)
    file.write(b0)
    #import pdb; pdb.set_trace()
    np.savez(msg, weight=wt_np, beta=beta.detach().cpu().numpy())
    #print(f'{msg}, {len(d)}, {beta}')

def b158_export(model, filepath):
    """
    Export the model weights in full float32 .bin file to be read from C.
    This is same as legacy_export, but with a proper header.
    """
    #import pdb; pdb.set_trace()
    version = 45400 #b158

    out_file = open(filepath, 'wb')
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "ak42" in ASCII
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) write version, which will be int
    out_file.write(struct.pack('i', version))
    # 3) write the params, which will be 7 ints
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) write some other flags
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    out_file.write(struct.pack('B', int(shared_classifier)))
    pad = 256 - out_file.tell() # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b'\0' * pad)

    print(version, shared_classifier)
    # now let's write out all the params
        # next write out the embedding weights
    serialize_fp32(out_file, model.tok_embeddings.weight, 'tok')

    # now all the layers
    # attention weights
    #for layer in model.layers:
    #    serialize_fp32(out_file, layer.attention_norm.weight)
    for ii, layer in enumerate(model.layers):
        serialize_b158(out_file, layer.attention.wq.weight, f'wq_{ii}')
    for ii, layer in enumerate(model.layers):
        serialize_b158(out_file, layer.attention.wk.weight, f'wk_{ii}')
    for ii, layer in enumerate(model.layers):
        serialize_b158(out_file, layer.attention.wv.weight, f'wv_{ii}')
    for ii, layer in enumerate(model.layers):
        serialize_b158(out_file, layer.attention.wo.weight, f'wo_{ii}')

    for ii, layer in enumerate(model.layers):
        serialize_fp32(out_file, layer.attention.wq.layernorm.weight, f'wq_n_{ii}')
    for ii, layer in enumerate(model.layers):
        serialize_fp32(out_file, layer.attention.wk.layernorm.weight, f'wk_n_{ii}')
    for ii, layer in enumerate(model.layers):
        serialize_fp32(out_file, layer.attention.wv.layernorm.weight, f'wv_n_{ii}')
    for ii, layer in enumerate(model.layers):
        serialize_fp32(out_file, layer.attention.wo.layernorm.weight, f'wo_n_{ii}')
    # ffn weights
    #for layer in model.layers:
    #    serialize_fp32(out_file, layer.ffn_norm.weight)
    for ii, layer in enumerate(model.layers):
        serialize_b158(out_file, layer.feed_forward.w1.weight, f'w1_{ii}')
    for ii, layer in enumerate(model.layers):
        serialize_b158(out_file, layer.feed_forward.w2.weight, f'w2_{ii}')
    for ii, layer in enumerate(model.layers):
        serialize_b158(out_file, layer.feed_forward.w3.weight, f'w3_{ii}')

    for ii, layer in enumerate(model.layers):
        serialize_fp32(out_file, layer.feed_forward.w1.layernorm.weight, f'w1_n_{ii}')
    for ii, layer in enumerate(model.layers):
        serialize_fp32(out_file, layer.feed_forward.w2.layernorm.weight, f'w2_n_{ii}')
    for ii, layer in enumerate(model.layers):
        serialize_fp32(out_file, layer.feed_forward.w3.layernorm.weight, f'w3_n_{ii}')
    # final rmsnorm
    #serialize_fp32(out_file, model.norm.weight)
    ## freqs_cis, 結局使ってないので削除
    #serialize_fp32(out_file, model.freqs_cos[:p.max_seq_len], 'fcos')
    #serialize_fp32(out_file, model.freqs_sin[:p.max_seq_len], 'fsin')

    # final classifier weights
    # tok_emmedding と share しているはずだけど、こっちは量子化しているので、取り敢えずセーブしておく
    serialize_fp32(out_file, model.output.layernorm.weight, f'out_n')
    serialize_b158(out_file, model.output.weight, f'out')
    #if not shared_classifier:
    #    serialize_b158(out_file, model.output.weight, f'out')

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")
