import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum PlaceholderWriter {
    /// Writes a deterministic gradient PNG so wiring and IO can be validated.
    static func writePlaceholderPNG(to url: URL, width: Int, height: Int, seed: Int, prompt: String) throws {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        var data = Data(count: height * bytesPerRow)

        // Simple deterministic pattern: depends on seed and prompt hash.
        var rng = SplitMix64(seed: UInt64(bitPattern: Int64(seed)) ^ UInt64(prompt.hashValue))
        data.withUnsafeMutableBytes { (ptr: UnsafeMutableRawBufferPointer) in
            guard let base = ptr.baseAddress else { return }
            for y in 0 ..< height {
                for x in 0 ..< width {
                    let offset = y * bytesPerRow + x * bytesPerPixel
                    let value = rng.nextByte()
                    base.advanced(by: offset + 0).storeBytes(of: value, as: UInt8.self) // R
                    base.advanced(by: offset + 1).storeBytes(of: UInt8((Int(value) + x) % 255), as: UInt8.self) // G
                    base.advanced(by: offset + 2).storeBytes(of: UInt8((Int(value) + y) % 255), as: UInt8.self) // B
                    base.advanced(by: offset + 3).storeBytes(of: 255, as: UInt8.self) // A
                }
            }
        }

        guard let provider = CGDataProvider(data: data as CFData) else {
            throw ZImageError.pipelineUnavailable(message: "Unable to create data provider")
        }

        let bitsPerPixel = bytesPerPixel * bitsPerComponent

        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bitsPerPixel: bitsPerPixel,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            throw ZImageError.pipelineUnavailable(message: "Unable to create CGImage")
        }

        let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        guard let dest else {
            throw ZImageError.pipelineUnavailable(message: "Unable to create PNG destination")
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        if !CGImageDestinationFinalize(dest) {
            throw ZImageError.pipelineUnavailable(message: "Failed to write PNG")
        }
    }
}

private struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    mutating func nextByte() -> UInt8 {
        UInt8(truncatingIfNeeded: next() >> 56)
    }
}
