import Foundation

public struct AspectPreset: Equatable, Sendable {
    public let name: String
    public let width: Int
    public let height: Int

    public init(_ name: String, _ width: Int, _ height: Int) {
        self.name = name
        self.width = width
        self.height = height
    }
}

public enum AspectPresets {
    public static let list: [AspectPreset] = [
        .init("1:1", 1024, 1024),
        .init("16:9", 1280, 720),
        .init("9:16", 720, 1280),
        .init("4:3", 1088, 816),
        .init("3:4", 816, 1088)
    ]

    public static func preset(named name: String) -> AspectPreset? {
        list.first { $0.name == name }
    }
}
