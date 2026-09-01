"""Geometry-grounded VQ-VAE (SceMoS port): a heightmap-conditioned decoder wrapped AROUND the
T2M-GPT HumanVQVAE, without forking T2M-GPT's model code.

Design (SceMoS: local heightmap into the DECODER, plain concatenation beat FiLM/cross-attn):
  encoder + quantizer are the pretrained/finetuned T2M-GPT modules, UNTOUCHED (so tokens keep
  meaning and encode() is unchanged). Between quantize and decode we fuse a per-frame local
  heightmap embedding into the quantized latent, then run the pretrained decoder.

  hm (bs,T,32,32) -> per-frame 2D-CNN -> (bs,T,C_h) -> (bs,C_h,T) -> adaptive-pool to the latent
  length T/4 -> concat onto x_quantized (bs,512,T/4) -> 1x1 fusion conv -> (bs,512,T/4) -> decoder.

WARM-START PRESERVATION (load-bearing): the fusion conv is initialised so its motion-channel
block is identity and its heightmap-channel block is zero, and biases are zero. So at init
fusion(concat[xq, hm_feat]) == xq EXACTLY -> the decoder sees the same input as the base model
-> reconstruction is bit-identical to the warm-start checkpoint before any training. The model
then LEARNS to use the heightmap from a known-good starting point (same discipline as
prepare_quantizer_for_finetune: start in the state of the converged model, then adapt).

A scene-less clip (H3D locomotion) passes a flat-floor heightmap (zeros) -> the decoder learns
"flat floor -> normal locomotion, raised surface -> sit/lie onto it".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

HM_N = 32


class HeightmapEncoder(nn.Module):
    """(bs, T, 32, 32) height-above-floor -> (bs, C_h, T) per-frame embedding. Small per-frame
    2D CNN (shared across frames), then flattened to a vector per frame."""

    def __init__(self, c_h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),      # 32->16
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),     # 16->8
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),     # 8->4
            nn.AdaptiveAvgPool2d(1),                    # 4->1
        )
        self.proj = nn.Linear(64, c_h)
        self.c_h = c_h

    def forward(self, hm):
        bs, t, h, w = hm.shape
        x = hm.reshape(bs * t, 1, h, w)
        x = self.net(x).reshape(bs * t, 64)
        x = self.proj(x).reshape(bs, t, self.c_h).permute(0, 2, 1)  # (bs, C_h, T)
        return x


class SceneVQVAE(nn.Module):
    """Wraps a loaded HumanVQVAE (`base`). forward(x, hm) reconstructs x conditioned on the local
    heightmap hm. encode() delegates to the base (tokens unchanged)."""

    def __init__(self, base, c_h=128):
        super().__init__()
        self.base = base                 # HumanVQVAE (encoder+quantizer+decoder), pretrained
        self.hm_encoder = HeightmapEncoder(c_h)
        emb = base.vqvae.code_dim        # 512
        self.fusion = nn.Conv1d(emb + c_h, emb, kernel_size=1)
        self._init_fusion_identity(emb, c_h)
        self.code_dim = emb

    def _init_fusion_identity(self, emb, c_h):
        with torch.no_grad():
            w = torch.zeros(emb, emb + c_h, 1)
            w[:, :emb, 0] = torch.eye(emb)      # pass x_quantized through
            w[:, emb:, 0] = 0.0                 # ignore heightmap at init
            self.fusion.weight.copy_(w)
            self.fusion.bias.zero_()

    # -- token extraction: identical to the base, heightmap-independent --
    def encode(self, x):
        return self.base.encode(x)

    def encode_latent(self, x):
        """Pre-quantization encoder latent (bs, code_dim, T/4). Used by the shift-consistency
        loss to force the encoder to be vertical-shift invariant (so tokens carry no absolute
        contact height -- the heightmap does)."""
        v = self.base.vqvae
        return v.encoder(v.preprocess(x))

    def forward_from_motion(self, x, hm):
        """Like forward, but also returns the pre-quant latent z (for the consistency loss)."""
        v = self.base.vqvae
        z = v.encoder(v.preprocess(x))
        x_quantized, loss, perplexity = v.quantizer(z)
        hm_feat = F.adaptive_avg_pool1d(self.hm_encoder(hm), x_quantized.shape[-1])
        fused = self.fusion(torch.cat([x_quantized, hm_feat], dim=1))
        return v.postprocess(v.decoder(fused)), loss, perplexity, z

    def forward(self, x, hm):
        v = self.base.vqvae
        x_in = v.preprocess(x)                       # (bs,263,T)->(bs,263,T)? actually (bs,T,263)->(bs,263,T)
        x_encoder = v.encoder(x_in)                  # (bs,512,T/4)
        x_quantized, loss, perplexity = v.quantizer(x_encoder)
        hm_feat = self.hm_encoder(hm)                # (bs,C_h,T)
        hm_feat = F.adaptive_avg_pool1d(hm_feat, x_quantized.shape[-1])  # -> (bs,C_h,T/4)
        fused = self.fusion(torch.cat([x_quantized, hm_feat], dim=1))    # (bs,512,T/4)
        x_decoder = v.decoder(fused)                 # (bs,263,T)
        x_out = v.postprocess(x_decoder)             # (bs,T,263)
        return x_out, loss, perplexity

    def forward_decoder(self, code_idx, hm):
        """Decode tokens conditioned on hm. code_idx: (1,T/4) longs. hm: (1,T,32,32)."""
        v = self.base.vqvae
        x_d = v.quantizer.dequantize(code_idx)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()  # (1,512,T/4)
        hm_feat = self.hm_encoder(hm)
        hm_feat = F.adaptive_avg_pool1d(hm_feat, x_d.shape[-1])
        fused = self.fusion(torch.cat([x_d, hm_feat], dim=1))
        x_decoder = v.decoder(fused)
        return v.postprocess(x_decoder)
