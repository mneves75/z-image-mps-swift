import Foundation
import MLX
import MLXNN

// MARK: - Core ops

struct RMSNorm {
    let weight: MLXArray
    let eps: Float
    func callAsFunction(_ x: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        let ms = MLX.mean(x * x, axes: [x.ndim - 1], keepDims: true, stream: stream)
        let inv = (ms + eps).sqrt(stream: stream).reciprocal(stream: stream)
        return x * inv * weight
    }
}

struct Linear {
    let weight: MLXArray
    let bias: MLXArray?

    func callAsFunction(_ x: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        let wT = weight.transposed(axes: [1, 0], stream: stream)
        var out = matmul(x, wT, stream: stream)
        if let b = bias { out = out + b }
        return out
    }
}

func silu(_ x: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    // Swish / SiLU activation with explicit stream to keep parity with MLX ops.
    x * sigmoid(x, stream: stream)
}

/// Rotate pairs in the last dimension: [-x_odd, x_even] interleaved (Llama/Qwen definition).
@inline(__always)
private func rotateHalf(_ x: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    let half = x.shape[3] / 2
    let x1 = x[0..., 0..., 0..., ..<half]
    let x2 = x[0..., 0..., 0..., half...]
    return concatenated([-x2, x1], axis: -1, stream: stream)
}

struct RotaryEmbedding {
    let theta: Float
    let headDim: Int
    let invFreq: MLXArray

    init(theta: Float, headDim: Int) {
        self.theta = theta
        self.headDim = headDim
        let exponents = (0..<(headDim / 2)).map { 2 * Float($0) / Float(headDim) }
        let invValues = exponents.map { pow(theta, -$0) }
        self.invFreq = MLXArray(invValues)
    }

    func apply(_ x: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        // x: [B, H, T, D]
        let seqLen = x.shape[2]
        let inv = invFreq.asType(.float32)
        let invExp = inv.reshaped([1, headDim / 2, 1], stream: stream) // [1, d/2, 1]
        let positions = MLXHelpers.arange(seqLen, dtype: .float32).reshaped([1, 1, seqLen], stream: stream) // [1,1,T]
        var freqs = matmul(invExp, positions, stream: stream).transposed(axes: [0, 2, 1], stream: stream) // [1, T, d/2]
        freqs = concatenated([freqs, freqs], axis: -1, stream: stream) // [1, T, d]
        let cos = freqs.cos(stream: stream).expandedDimensions(axis: 1, stream: stream) // [1,1,T,d]
        let sin = freqs.sin(stream: stream).expandedDimensions(axis: 1, stream: stream)
        return x * cos + rotateHalf(x, stream: stream) * sin
    }
}

struct GatedMLP {
    let gate: Linear
    let up: Linear
    let down: Linear

    func callAsFunction(_ x: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        let g = silu(gate(x, stream: stream), stream: stream)
        let u = up(x, stream: stream)
        return down(g * u, stream: stream)
    }
}

struct MultiHeadAttention {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let qProj: Linear
    let kProj: Linear
    let vProj: Linear
    let oProj: Linear
    let qNorm: RMSNorm
    let kNorm: RMSNorm
    let rotary: RotaryEmbedding

    func callAsFunction(_ x: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        let q = qProj(x, stream: stream)
        let k = kProj(x, stream: stream)
        let v = vProj(x, stream: stream)

        var qh = q.reshaped([q.shape[0], q.shape[1], numHeads, headDim], stream: stream).transposed(axes: [0, 2, 1, 3], stream: stream)
        var kh = k.reshaped([k.shape[0], k.shape[1], numKVHeads, headDim], stream: stream).transposed(axes: [0, 2, 1, 3], stream: stream)

        // Per-head RMSNorm on q and k as in Qwen3 reference.
        qh = qNorm(qh, stream: stream)
        kh = kNorm(kh, stream: stream)
        let vh = v.reshaped([v.shape[0], v.shape[1], numKVHeads, headDim], stream: stream).transposed(axes: [0, 2, 1, 3], stream: stream)

        let qRot = rotary.apply(qh, stream: stream)
        let kRot = rotary.apply(kh, stream: stream)

        let repeatFactor = numHeads / numKVHeads
        // repeat_kv: repeat each KV head `repeatFactor` times, preserving head grouping order
        let kExp: MLXArray
        let vExp: MLXArray
        if repeatFactor == 1 {
            kExp = kRot
            vExp = vh
        } else {
            let kExpanded = kRot.expandedDimensions(axis: 2, stream: stream) // [B, numKV, 1, T, D]
            let vExpanded = vh.expandedDimensions(axis: 2, stream: stream)
            let kRepeated = MLXHelpers.repeatArray(kExpanded, count: repeatFactor, axis: 2, stream: stream)
            let vRepeated = MLXHelpers.repeatArray(vExpanded, count: repeatFactor, axis: 2, stream: stream)
            kExp = kRepeated.reshaped([kRot.shape[0], numKVHeads * repeatFactor, kRot.shape[2], headDim], stream: stream)
            vExp = vRepeated.reshaped([vh.shape[0], numKVHeads * repeatFactor, vh.shape[2], headDim], stream: stream)
        }

        var attn = matmul(qRot, kExp.transposed(axes: [0, 1, 3, 2], stream: stream), stream: stream) / Float(headDim).squareRoot()

        // Causal mask: add -inf to disallow attending to future positions (matches HF additive mask semantics).
        let seqLen = qRot.shape[2]
        let positions = MLXHelpers.arange(seqLen, dtype: .float32)
        let i = positions.reshaped([1, 1, seqLen, 1], stream: stream)
        let j = positions.reshaped([1, 1, 1, seqLen], stream: stream)
        let future = greater(j, i, stream: stream)
        let mask = MLX.where(future, MLXArray(-Float.infinity), MLXArray(0.0), stream: stream)
        attn = attn + mask

        let probs = softmax(attn, axis: -1, stream: stream)
        let ctx = matmul(probs, vExp, stream: stream) // [B, H, T, D]
        let merged = ctx.transposed(axes: [0, 2, 1, 3], stream: stream).reshaped([x.shape[0], x.shape[1], numHeads * headDim], stream: stream)
        return oProj(merged, stream: stream)
    }
}

struct QwenBlock {
    let attnNorm: RMSNorm
    let attn: MultiHeadAttention
    let mlpNorm: RMSNorm
    let mlp: GatedMLP

    func callAsFunction(_ x: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        let h1 = x + attn(attnNorm(x, stream: stream), stream: stream)
        let h2 = h1 + mlp(mlpNorm(h1, stream: stream), stream: stream)
        return h2
    }
}

public final class Qwen3Encoder {
    private let embed: MLXArray
    private let layers: [QwenBlock]
    private let finalNorm: RMSNorm
    private let config: TextEncoderConfig

    public init(weights: [String: MLXArray], config: TextEncoderConfig) {
        self.config = config
        self.embed = weights["model.embed_tokens.weight"]!.asType(.float32)

        let rotary = RotaryEmbedding(theta: 1_000_000.0, headDim: config.head_dim)
        var blocks: [QwenBlock] = []
        for i in 0..<config.num_hidden_layers {
            let p = "model.layers.\(i)."
            let attnNorm = RMSNorm(weight: weights[p + "input_layernorm.weight"]!.asType(.float32), eps: config.rms_norm_eps)
            let mlpNorm = RMSNorm(weight: weights[p + "post_attention_layernorm.weight"]!.asType(.float32), eps: config.rms_norm_eps)
            let qProj = Linear(weight: weights[p + "self_attn.q_proj.weight"]!.asType(.float32), bias: nil)
            let kProj = Linear(weight: weights[p + "self_attn.k_proj.weight"]!.asType(.float32), bias: nil)
            let vProj = Linear(weight: weights[p + "self_attn.v_proj.weight"]!.asType(.float32), bias: nil)
            let oProj = Linear(weight: weights[p + "self_attn.o_proj.weight"]!.asType(.float32), bias: nil)
            let qNorm = RMSNorm(weight: weights[p + "self_attn.q_norm.weight"]!.asType(.float32), eps: config.rms_norm_eps)
            let kNorm = RMSNorm(weight: weights[p + "self_attn.k_norm.weight"]!.asType(.float32), eps: config.rms_norm_eps)
            let attn = MultiHeadAttention(
                numHeads: config.num_attention_heads,
                numKVHeads: config.num_key_value_heads,
                headDim: config.head_dim,
                qProj: qProj,
                kProj: kProj,
                vProj: vProj,
                oProj: oProj,
                qNorm: qNorm,
                kNorm: kNorm,
                rotary: rotary
            )
            let gate = Linear(weight: weights[p + "mlp.gate_proj.weight"]!.asType(.float32), bias: nil)
            let up = Linear(weight: weights[p + "mlp.up_proj.weight"]!.asType(.float32), bias: nil)
            let down = Linear(weight: weights[p + "mlp.down_proj.weight"]!.asType(.float32), bias: nil)
            let mlp = GatedMLP(gate: gate, up: up, down: down)
            blocks.append(QwenBlock(attnNorm: attnNorm, attn: attn, mlpNorm: mlpNorm, mlp: mlp))
        }
        self.layers = blocks
        self.finalNorm = RMSNorm(weight: weights["model.norm.weight"]!.asType(.float32), eps: config.rms_norm_eps)
    }

    public func encode(ids: [Int], stream: StreamOrDevice = .default) -> MLXArray {
        let idsArray = MLXArray(ids.map { Int32($0) })
        var x = embed.take(idsArray, axis: 0, stream: stream).asType(.float32).expandedDimensions(axis: 0, stream: stream) // [1,T,C]
        for layer in layers {
            x = layer(x, stream: stream)
        }
        return finalNorm(x, stream: stream)
    }

    /// Expose intermediate layer output for testing.
    public func encodeLayer(ids: [Int], layerIndex: Int, stream: StreamOrDevice = .default) -> MLXArray {
        precondition(layerIndex < layers.count)
        let idsArray = MLXArray(ids.map { Int32($0) })
        var x = embed.take(idsArray, axis: 0, stream: stream).asType(.float32).expandedDimensions(axis: 0, stream: stream)
        for i in 0...layerIndex {
            x = layers[i](x, stream: stream)
        }
        return x
    }
}
