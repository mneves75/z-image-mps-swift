import Foundation
import MLX

/// Loads sharded safetensors using the HF index file.
public enum WeightLoader {
    public static func loadShardedSafetensors(at directory: URL, indexFile: String) throws -> [String: MLXArray] {
        let indexURL = directory.appendingPathComponent(indexFile)
        let data = try Data(contentsOf: indexURL)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        guard let weightMap = json["weight_map"] as? [String: String] else {
            throw ZImageError.weightsCorrupted("weight_map missing in index")
        }
        // Group params by shard file
        var perFile: [String: [String]] = [:]
        for (name, file) in weightMap {
            perFile[file, default: []].append(name)
        }

        var merged: [String: MLXArray] = [:]
        for (file, _) in perFile {
            let shardURL = directory.appendingPathComponent(file)
            let shard = try WeightsRepository.loadSafetensors(at: shardURL)
            for (k, v) in shard {
                merged[k] = v
            }
        }
        return merged
    }
}
