import Foundation
import Testing
@testable import ZImageCore

struct ConfigTests {
    @Test
    func loadTextEncoderConfig() throws {
        let cfg = try TextEncoderConfig.load(from: URL(fileURLWithPath: ".cache/hf/Z-Image-Turbo/text_encoder/config.json"))
        #expect(cfg.hidden_size == 2560)
        #expect(cfg.num_hidden_layers == 36)
        #expect(cfg.num_attention_heads == 32)
        #expect(cfg.num_key_value_heads == 8)
        #expect(cfg.head_dim == 128)
    }

    @Test
    func loadTransformerConfig() throws {
        let cfg = try TransformerConfig.load(from: URL(fileURLWithPath: ".cache/hf/Z-Image-Turbo/transformer/config.json"))
        #expect(cfg.dim == 3840)
        #expect(cfg.n_layers == 30)
        #expect(cfg.n_heads == 30)
        #expect(cfg.n_kv_heads == 30)
        #expect(cfg.in_channels == 16)
    }

    @Test
    func loadVAEConfig() throws {
        let cfg = try VAEConfig.load(from: URL(fileURLWithPath: ".cache/hf/Z-Image-Turbo/vae/config.json"))
        #expect(cfg.latent_channels == 16)
        #expect(cfg.block_out_channels == [128, 256, 512, 512])
        #expect(cfg.scaling_factor > 0)
    }
}
