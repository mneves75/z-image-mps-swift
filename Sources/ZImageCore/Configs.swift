import Foundation

public struct TextEncoderConfig: Codable {
    public let hidden_size: Int
    public let num_hidden_layers: Int
    public let num_attention_heads: Int
    public let num_key_value_heads: Int
    public let intermediate_size: Int
    public let rms_norm_eps: Float
    public let head_dim: Int

    enum CodingKeys: String, CodingKey {
        case hidden_size
        case num_hidden_layers
        case num_attention_heads
        case num_key_value_heads
        case intermediate_size
        case rms_norm_eps = "rms_norm_eps"
        case head_dim
    }

    public static func load(from url: URL) throws -> TextEncoderConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(TextEncoderConfig.self, from: data)
    }
}

public struct TransformerConfig: Codable {
    public let dim: Int
    public let n_layers: Int
    public let n_heads: Int
    public let n_kv_heads: Int
    public let in_channels: Int
    public let all_patch_size: [Int]
    public let rope_theta: Float
    public let qk_norm: Bool
    public let n_refiner_layers: Int

    enum CodingKeys: String, CodingKey {
        case dim
        case n_layers
        case n_heads
        case n_kv_heads
        case in_channels
        case all_patch_size
        case rope_theta
        case qk_norm
        case n_refiner_layers
    }

    public static func load(from url: URL) throws -> TransformerConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(TransformerConfig.self, from: data)
    }
}

public struct VAEConfig: Codable {
    public let in_channels: Int
    public let out_channels: Int
    public let latent_channels: Int
    public let block_out_channels: [Int]
    public let layers_per_block: Int
    public let scaling_factor: Float
    public let shift_factor: Float
    public let force_upcast: Bool

    enum CodingKeys: String, CodingKey {
        case in_channels
        case out_channels
        case latent_channels
        case block_out_channels
        case layers_per_block
        case scaling_factor
        case shift_factor
        case force_upcast
    }

    public static func load(from url: URL) throws -> VAEConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(VAEConfig.self, from: data)
    }
}
