import Foundation

public struct GenerationRequest: Sendable {
    public let prompt: String
    public let negativePrompt: String?
    public let width: Int
    public let height: Int
    public let steps: Int
    public let guidance: Double
    public let seed: Int

    public init(prompt: String,
                negativePrompt: String?,
                width: Int,
                height: Int,
                steps: Int,
                guidance: Double,
                seed: Int) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
    }
}

public struct GenerationResult: Sendable {
    public let imagePath: URL
    public let metadata: [String: String]
}

public protocol ImageGenerator {
    func generate(_ request: GenerationRequest, outputURL: URL) async throws -> GenerationResult
}

/// Temporary stub generator; replace with MLX pipeline.
public final class StubImageGenerator: ImageGenerator {
    public init() {}

    public func generate(_ request: GenerationRequest, outputURL: URL) async throws -> GenerationResult {
        try DimensionValidator.validate(width: request.width, height: request.height)

        // Save a tiny placeholder PNG to prove plumbing; replace with MLX output later.
        try PlaceholderWriter.writePlaceholderPNG(
            to: outputURL,
            width: request.width,
            height: request.height,
            seed: request.seed,
            prompt: request.prompt
        )

        let meta: [String: String] = [
            "width": "\(request.width)",
            "height": "\(request.height)",
            "seed": "\(request.seed)",
            "steps": "\(request.steps)",
            "guidance": "\(request.guidance)"
        ]
        return GenerationResult(imagePath: outputURL, metadata: meta)
    }
}

/// Generator that validates weights manifest then defers to stub output for now.
/// Keeps integrity checks centralized while MLX pipeline is completed.
public final class VerifiedPlaceholderGenerator: ImageGenerator {
    private let weightsDir: URL

    public init(weightsDir: URL) {
        self.weightsDir = weightsDir
    }

    public func generate(_ request: GenerationRequest, outputURL: URL) async throws -> GenerationResult {
        _ = try WeightsRepository.verify(at: weightsDir)
        try WeightsRepository.requireComponents(in: weightsDir)
        let stub = StubImageGenerator()
        return try await stub.generate(request, outputURL: outputURL)
    }
}
