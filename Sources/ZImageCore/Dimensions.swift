import Foundation

public struct DimensionPolicy {
    public let multiple: Int
    public let min: Int
    public let max: Int

    public init(multiple: Int = 64, min: Int = 256, max: Int = 2048) {
        self.multiple = multiple
        self.min = min
        self.max = max
    }
}

public enum DimensionValidator {
    public static func validate(width: Int, height: Int, policy: DimensionPolicy = .init()) throws {
        guard width >= policy.min, height >= policy.min else {
            throw ZImageError.invalidDimensions(width: width, height: height, reason: "below minimum \(policy.min)")
        }
        guard width <= policy.max, height <= policy.max else {
            throw ZImageError.invalidDimensions(width: width, height: height, reason: "above maximum \(policy.max)")
        }
        guard width % policy.multiple == 0, height % policy.multiple == 0 else {
            throw ZImageError.invalidDimensions(width: width, height: height, reason: "must be multiples of \(policy.multiple)")
        }
    }
}
