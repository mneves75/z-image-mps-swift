import Foundation
import ArgumentParser

public enum ExecutionDevice: String, CaseIterable, Sendable, Codable, ExpressibleByArgument {
    case auto
    case metal
    case cpu
}

public struct DeviceSelection {
    public let device: ExecutionDevice
    public let reason: String
}

public enum DeviceSelector {
    /// For now we cannot directly query MLX; assume Metal available on Apple Silicon.
    public static func select(preferred: ExecutionDevice, allowCPU: Bool) throws -> DeviceSelection {
        switch preferred {
        case .auto:
            return DeviceSelection(device: .metal, reason: "auto→metal")
        case .metal:
            return DeviceSelection(device: .metal, reason: "metal requested")
        case .cpu:
            guard allowCPU else { throw ZImageError.cpuNotAllowed }
            return DeviceSelection(device: .cpu, reason: "cpu requested")
        }
    }
}
