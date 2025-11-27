import Foundation
import Testing
import MLX
@testable import ZImageCore

struct Qwen3Tests {
    @Test
    func encoderStatsMatchFixture() throws {
        let fixtureURL = URL(fileURLWithPath: "Tests/fixtures/zimage_model_fixtures.json")
        let data = try Data(contentsOf: fixtureURL)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let textHidden = json["text_hidden"] as! [String: Any]
        let expShape = textHidden["shape"] as! [Int]
        let expMean = textHidden["mean"] as! Double
        let expStd = textHidden["std"] as! Double
        let expSlice = textHidden["slice"] as! [[Double]]
        let ids = Array((json["token_ids_32"] as! [Int]).prefix(expShape[1]))

        let weights = try WeightLoader.loadShardedSafetensors(
            at: URL(fileURLWithPath: ".cache/hf/Z-Image-Turbo/text_encoder"),
            indexFile: "model.safetensors.index.json"
        )
        let cfg = try TextEncoderConfig.load(from: URL(fileURLWithPath: ".cache/hf/Z-Image-Turbo/text_encoder/config.json"))
        let encoder = Qwen3Encoder(weights: weights, config: cfg)
        let out = encoder.encode(ids: ids, stream: .cpu)

        #expect(out.shape == expShape)
        let mean = out.mean().item(Double.self)
        let stdVal = MLX.std(out).item(Double.self)
        #expect(abs(mean - expMean) < 5e-3)
        #expect(abs(stdVal - expStd) < 5e-3)

        let slice = out[0, 0..<4, 0..<4]
        var sliceArray = Array(repeating: Array(repeating: 0.0, count: 4), count: 4)
        for i in 0..<4 {
            for j in 0..<4 {
                sliceArray[i][j] = slice[i, j].item(Double.self)
            }
        }
        for i in 0..<4 {
            for j in 0..<4 {
                #expect(abs(sliceArray[i][j] - expSlice[i][j]) < 5e-2)
            }
        }
    }

    @Test
    func firstLayerMatchesFixture() throws {
        let layerURL = URL(fileURLWithPath: "Tests/fixtures/text_layer0.json")
        let data = try Data(contentsOf: layerURL)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let expMean = json["mean"] as! Double
        let expStd = json["std"] as! Double
        let expSlice = json["slice"] as! [[Double]]

        let baseFixture = URL(fileURLWithPath: "Tests/fixtures/zimage_model_fixtures.json")
        let baseData = try Data(contentsOf: baseFixture)
        let baseJson = try JSONSerialization.jsonObject(with: baseData) as! [String: Any]
        let ids = Array((baseJson["token_ids_32"] as! [Int]).prefix(13))

        let weights = try WeightLoader.loadShardedSafetensors(
            at: URL(fileURLWithPath: ".cache/hf/Z-Image-Turbo/text_encoder"),
            indexFile: "model.safetensors.index.json"
        )
        let cfg = try TextEncoderConfig.load(from: URL(fileURLWithPath: ".cache/hf/Z-Image-Turbo/text_encoder/config.json"))
        let encoder = Qwen3Encoder(weights: weights, config: cfg)
        let layer0 = encoder.encodeLayer(ids: ids, layerIndex: 0, stream: .cpu) // custom helper

        let mean = layer0.mean().item(Double.self)
        let stdVal = MLX.std(layer0).item(Double.self)
        #expect(abs(mean - expMean) < 5e-3)
        #expect(abs(stdVal - expStd) < 5e-3)

        let slice = layer0[0, 0..<4, 0..<4]
        var sliceArray = Array(repeating: Array(repeating: 0.0, count: 4), count: 4)
        for i in 0..<4 {
            for j in 0..<4 {
                sliceArray[i][j] = slice[i, j].item(Double.self)
            }
        }
        for i in 0..<4 {
            for j in 0..<4 {
                #expect(abs(sliceArray[i][j] - expSlice[i][j]) < 5e-2)
            }
        }
    }
}
