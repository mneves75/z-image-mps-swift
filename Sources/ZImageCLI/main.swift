import Foundation
import ArgumentParser
import ZImageCore

@main
struct ZImageCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "z-image-cli",
        abstract: "Run Z-Image Turbo locally via MLX (placeholder pipeline for now)."
    )

    @Option(name: [.customShort("p"), .long], help: "Text prompt.")
    var prompt: String = "Analog film portrait of a skateboarder, shallow depth of field"

    @Option(name: .long, help: "Negative prompt.")
    var negativePrompt: String?

    @Option(name: [.customShort("s"), .long], help: "Number of inference steps.")
    var steps: Int = 9

    @Option(name: .long, help: "CFG guidance scale (Turbo expects ~0.0).")
    var guidanceScale: Double = 0.0

    @Option(name: .long, help: "Height in pixels (ignored when --aspect is set).")
    var height: Int = 1024

    @Option(name: .long, help: "Width in pixels (ignored when --aspect is set).")
    var width: Int = 1024

    @Option(name: .long, help: "Aspect preset: \(AspectPresets.list.map { $0.name }.joined(separator: ","))")
    var aspect: String?

    @Option(name: .long, help: "Seed for reproducibility (increments per image).")
    var seed: Int?

    @Option(name: .long, help: "Number of images to generate.")
    var numImages: Int = 1

    @Option(name: [.customShort("o"), .long], help: "Output file or directory.")
    var output: String?

    @Option(name: .long, help: "Directory for outputs (ignored when --output is file).")
    var outdir: String?

    @Option(name: .long, help: "Device: auto | metal | cpu")
    var device: ExecutionDevice = .auto

    @Flag(name: .long, help: "Permit CPU execution (off by default).")
    var allowCPU: Bool = false

    @Option(name: .long, help: "Weights directory containing manifest.json and safetensors; enables integrity check.")
    var weights: String?

    mutating func run() async throws {
        let selection = try DeviceSelector.select(preferred: device, allowCPU: allowCPU)
        print("Device: \(selection.device.rawValue) (\(selection.reason))")

        let dims = try resolvedDimensions()
        let seeds = SeedGenerator.seeds(count: numImages, baseSeed: seed)

        let generator: ImageGenerator
        if let weights {
            let dir = URL(fileURLWithPath: NSString(string: weights).expandingTildeInPath, isDirectory: true)
            generator = VerifiedPlaceholderGenerator(weightsDir: dir)
        } else {
            generator = StubImageGenerator()
        }
        let start = Date()

        for (index, seed) in seeds.enumerated() {
            let outputURL = try PathResolver.resolveOutput(output: output, outdir: outdir, index: index)
            let request = GenerationRequest(
                prompt: prompt,
                negativePrompt: negativePrompt,
                width: dims.width,
                height: dims.height,
                steps: steps,
                guidance: guidanceScale,
                seed: seed
            )
            let tick = Date()
            print("[\(index + 1)/\(seeds.count)] seed=\(seed) size=\(dims.width)x\(dims.height) steps=\(steps) guidance=\(guidanceScale) -> \(outputURL.path)")
            let result = try await generator.generate(request, outputURL: outputURL)
            let elapsed = Date().timeIntervalSince(tick)
            print(String(format: "Saved %@ in %.2fs", result.imagePath.path, elapsed))
        }

        let total = Date().timeIntervalSince(start)
        print(String(format: "Done in %.2fs", total))
    }

    private func resolvedDimensions() throws -> (width: Int, height: Int) {
        if let aspect {
            guard let preset = AspectPresets.preset(named: aspect) else {
                throw ValidationError("Unknown aspect \(aspect)")
            }
            try DimensionValidator.validate(width: preset.width, height: preset.height)
            return (preset.width, preset.height)
        }
        try DimensionValidator.validate(width: width, height: height)
        return (width, height)
    }
}
