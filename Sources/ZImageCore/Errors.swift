import Foundation

public enum ZImageError: Error, CustomStringConvertible {
    case invalidDimensions(width: Int, height: Int, reason: String)
    case unsupportedDevice(String)
    case weightsMissing(URL)
    case weightsCorrupted(String)
    case integrityMismatch(path: URL)
    case cpuNotAllowed
    case pipelineUnavailable(message: String)

    public var description: String {
        switch self {
        case let .invalidDimensions(width, height, reason):
            return "Invalid dimensions \(width)x\(height): \(reason)"
        case let .unsupportedDevice(message):
            return "Unsupported device: \(message)"
        case let .weightsMissing(url):
            return "Required weights not found at \(url.path)"
        case let .integrityMismatch(path):
            return "Integrity check failed for \(path.path)"
        case let .weightsCorrupted(message):
            return "Weights corrupted: \(message)"
        case .cpuNotAllowed:
            return "CPU execution is disabled; pass --allow-cpu to override."
        case let .pipelineUnavailable(message):
            return "Pipeline unavailable: \(message)"
        }
    }
}
