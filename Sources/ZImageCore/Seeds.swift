import Foundation

public enum SeedGenerator {
    /// Deterministic sequence: baseSeed + index; if baseSeed nil, random 63-bit.
    public static func seeds(count: Int, baseSeed: Int?) -> [Int] {
        let count = max(1, count)
        if let base = baseSeed {
            return (0 ..< count).map { base + $0 }
        } else {
            return (0 ..< count).map { _ in Int.random(in: 0 ..< Int.max >> 1) }
        }
    }
}
