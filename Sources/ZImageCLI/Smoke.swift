import ArgumentParser
import Foundation
import ZImageCore
import MLX

/// Minimal smoke check: load cached text encoder, run prompt -> stats (CPU) to validate weights.
struct Smoke: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Run a fast text-encoder smoke test (no image generation).")

    @Option(name: .long, help: "Weights root (expects text_encoder dir + config + shards).")
    var weights: String = "~/.cache/z-image-mlx/converted"

    @Option(name: .long, help: "Prompt text to encode.")
    var prompt: String = "smoke test"

    func run() throws {
        let root = URL(fileURLWithPath: NSString(string: weights).expandingTildeInPath)
        let teDir = root.appendingPathComponent("text_encoder")

        // Load config and weights
        let cfg = try TextEncoderConfig.load(from: teDir.appendingPathComponent("config.json"))
        let arrays = try WeightLoader.loadShardedSafetensors(
            at: teDir,
            indexFile: "model.safetensors.index.json"
        )
        let encoder = Qwen3Encoder(weights: arrays, config: cfg)

        // Load tokenizer files from root
        let tokenizer = try QwenTokenizer(
            vocabURL: root.appendingPathComponent("vocab.json"),
            mergesURL: root.appendingPathComponent("merges.txt")
        )
        let ids = tokenizer.encode(prompt)
        let hidden = encoder.encode(ids: ids, stream: .cpu)
        let mean = hidden.mean().item(Double.self)
        let std = MLX.std(hidden).item(Double.self)
        print(String(format: "Smoke OK — tokens %d | mean %.6f | std %.6f", ids.count, mean, std))
    }
}
