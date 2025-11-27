import Foundation
import Testing
import MLX
@testable import ZImageCore
@testable import ZImageTools

struct ConverterTests {
    @Test
    func convertsSafetensorsAndManifest() throws {
        let fm = FileManager.default
        let src = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
        let dst = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
        try fm.createDirectory(at: src, withIntermediateDirectories: true)
        try fm.createDirectory(at: dst, withIntermediateDirectories: true)

        // Build dummy safetensors files
        let arr = MLXArray.ones([2, 2])
        let components = ["text_encoder.safetensors", "vae.safetensors", "transformer.safetensors"]
        for name in components {
            let url = src.appendingPathComponent(name)
            try save(arrays: ["x": arr], metadata: [:], url: url)
        }
        // Tokenizer/config files
        try "{}".data(using: .utf8)!.write(to: src.appendingPathComponent("model_index.json"))
        try "{}".data(using: .utf8)!.write(to: src.appendingPathComponent("tokenizer_config.json"))
        try "".data(using: .utf8)!.write(to: src.appendingPathComponent("vocab.json"))
        try "".data(using: .utf8)!.write(to: src.appendingPathComponent("merges.txt"))

        // Run converter logic
        let dtype: DType = .float16
        for name in components {
            let srcURL = src.appendingPathComponent(name)
            let dstURL = dst.appendingPathComponent(name)
            try WeightsRepository.convertSafetensors(input: srcURL, output: dstURL, dtype: dtype)
        }
        // Copy config/tokenizer files as convert command would
        for name in ["model_index.json", "tokenizer_config.json", "vocab.json", "merges.txt"] {
            let s = src.appendingPathComponent(name)
            let d = dst.appendingPathComponent(name)
            try Data(contentsOf: s).write(to: d)
        }
        let manifest = try ManifestBuilder.build(for: dst, relativeTo: dst)
        #expect(manifest.files.count == components.count + 4)

        // Ensure dtype cast by loading back one file
        let back = try WeightsRepository.loadSafetensors(at: dst.appendingPathComponent("vae.safetensors"))
        #expect(back["x"]?.dtype == dtype)
    }
}
