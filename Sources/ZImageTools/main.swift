import ArgumentParser
import Foundation
import ZImageCore
import MLX

@main
struct ZImageTools: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "z-image-tools",
        abstract: "Utilities for fetching and converting Z-Image weights.",
        subcommands: [Fetch.self, Convert.self, ManifestCmd.self, Bench.self],
        defaultSubcommand: Fetch.self
    )
}

struct Fetch: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Fetch model weights from Hugging Face.")

    @Option(name: .long, help: "Destination directory for raw Hugging Face files.")
    var output: String = "~/.cache/z-image-mlx/raw"

    @Option(name: .long, help: "Model repo (HF).")
    var repo: String = "Tongyi-MAI/Z-Image-Turbo"

    func run() throws {
        let expanded = NSString(string: output).expandingTildeInPath
        try FileManager.default.createDirectory(atPath: expanded, withIntermediateDirectories: true)
        let cmd = "huggingface-cli download \(repo) --local-dir \(expanded) --include \"*\" --resume-download"
        print("→ \(cmd)")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-lc", cmd]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ValidationError("huggingface-cli failed with status \(process.terminationStatus)")
        }
        print("Download completed into \(expanded)")
    }
}

struct Convert: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Convert weights to MLX format.")

    @Option(name: .long, help: "Source directory containing downloaded model files.")
    var source: String = "~/.cache/z-image-mlx/raw"

    @Option(name: .long, help: "Destination directory for converted MLX weights.")
    var output: String = "~/.cache/z-image-mlx/converted"

    @Option(name: .long, help: "Output dtype for tensors (bf16|fp16|fp32).")
    var dtype: String = "bf16"

    func run() throws {
        let src = URL(fileURLWithPath: NSString(string: source).expandingTildeInPath)
        let dst = URL(fileURLWithPath: NSString(string: output).expandingTildeInPath, isDirectory: true)
        try FileManager.default.createDirectory(at: dst, withIntermediateDirectories: true)
        let dtypeValue: DType
        switch dtype.lowercased() {
        case "bf16", "bfloat16": dtypeValue = .bfloat16
        case "fp16", "float16": dtypeValue = .float16
        case "fp32", "float32": dtypeValue = .float32
        default: throw ValidationError("Unsupported dtype \(dtype)")
        }

        try WeightsRepository.requireComponents(in: src)

        let converterMap = [
            "text_encoder.safetensors",
            "vae.safetensors",
            "transformer.safetensors"
        ]

        for file in converterMap {
            let srcURL = src.appendingPathComponent(file)
            let dstURL = dst.appendingPathComponent(file)
            try WeightsRepository.convertSafetensors(input: srcURL, output: dstURL, dtype: dtypeValue)
            print("Converted \(file) -> \(dtypeValue)")
        }

        // Copy configs/tokenizer files untouched.
        let passthrough = ["model_index.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
        for name in passthrough {
            let s = src.appendingPathComponent(name)
            let d = dst.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: d.path) { try FileManager.default.removeItem(at: d) }
            try FileManager.default.copyItem(at: s, to: d)
        }

        let manifest = try ManifestBuilder.build(for: dst, relativeTo: dst)
        let manifestURL = dst.appendingPathComponent("manifest.json")
        let data = try JSONEncoder().encode(manifest)
        try data.write(to: manifestURL, options: .atomic)
        print("Converted weights and wrote manifest at \(manifestURL.path)")
    }
}

struct ManifestCmd: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Generate integrity manifest for a directory.")

    @Option(name: .long, help: "Directory to scan.")
    var directory: String

    func run() throws {
        let url = URL(fileURLWithPath: NSString(string: directory).expandingTildeInPath)
        let manifest = try ManifestBuilder.build(for: url, relativeTo: url)
        let data = try JSONEncoder().encode(manifest)
        let out = url.appendingPathComponent("manifest.json")
        try data.write(to: out, options: .atomic)
        print("Wrote manifest: \(out.path) (\(manifest.files.count) files)")
    }
}

struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Quick benchmark of stub pipeline IO.")

    @Option(name: .long, help: "Iterations.")
    var iterations: Int = 3

    func run() async throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("zimage-bench")
        try? FileManager.default.removeItem(at: tmp)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        let generator = StubImageGenerator()
        let dims = (width: 512, height: 512)
        let seeds = SeedGenerator.seeds(count: iterations, baseSeed: 1234)
        let start = Date()
        for (idx, seed) in seeds.enumerated() {
            let url = tmp.appendingPathComponent(OutputNamer.filename(index: idx, base: "bench"))
            let req = GenerationRequest(prompt: "bench", negativePrompt: nil, width: dims.width, height: dims.height, steps: 1, guidance: 0, seed: seed)
            _ = try await generator.generate(req, outputURL: url)
        }
        let total = Date().timeIntervalSince(start)
        print(String(format: "Bench wrote %d images to %@ in %.2fs", iterations, tmp.path, total))
    }
}
