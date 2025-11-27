import Foundation
import MLX
import MLXNN

public enum WeightsRepository {
    public static func verify(at directory: URL, manifestName: String = "manifest.json") throws -> Manifest {
        let manifestURL = directory.appendingPathComponent(manifestName)
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ZImageError.weightsMissing(manifestURL)
        }
        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        for file in manifest.files {
            let path = directory.appendingPathComponent(file.path)
            guard FileManager.default.fileExists(atPath: path.path) else {
                throw ZImageError.weightsMissing(path)
            }
            let actual = try FileHasher.sha256(url: path)
            if actual != file.sha256 {
                throw ZImageError.integrityMismatch(path: path)
            }
        }
        return manifest
    }

    public static func loadSafetensors(at path: URL) throws -> [String: MLXArray] {
        return try loadArrays(url: path)
    }

    /// Convert safetensors to a new safetensors file with unified dtype (default bf16).
    public static func convertSafetensors(input: URL, output: URL, dtype: DType = .bfloat16) throws {
        let arrays = try loadArrays(url: input)
        let casted = arrays.mapValues { $0.asType(dtype) }
        try save(arrays: casted, metadata: [:], url: output)
    }

    /// Ensures key components exist before attempting to build the pipeline.
    /// We deliberately keep this strict to fail fast on incomplete downloads.
    public static func requireComponents(in directory: URL) throws {
        let required = [
            "text_encoder.safetensors",
            "vae.safetensors",
            "transformer.safetensors",
            "model_index.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt"
        ]
        for filename in required {
            let path = directory.appendingPathComponent(filename)
            guard FileManager.default.fileExists(atPath: path.path) else {
                throw ZImageError.weightsMissing(path)
            }
        }
    }
}
