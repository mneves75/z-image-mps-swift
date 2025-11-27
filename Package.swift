// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "z-image-mlx",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "ZImageCore", targets: ["ZImageCore"]),
        .executable(name: "z-image-cli", targets: ["ZImageCLI"]),
        .executable(name: "z-image-tools", targets: ["ZImageTools"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.4.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.29.1")
    ],
    targets: [
        .target(
            name: "ZImageCore",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift")
            ],
            swiftSettings: [
                .unsafeFlags(["-warnings-as-errors"])
            ]
        ),
        .executableTarget(
            name: "ZImageCLI",
            dependencies: [
                "ZImageCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            swiftSettings: [
                .unsafeFlags(["-warnings-as-errors"])
            ]
        ),
        .executableTarget(
            name: "ZImageTools",
            dependencies: [
                "ZImageCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            swiftSettings: [
                .unsafeFlags(["-warnings-as-errors"])
            ]
        ),
        .testTarget(
            name: "ZImageTests",
            dependencies: ["ZImageCore"],
            swiftSettings: [
                .unsafeFlags(["-warnings-as-errors"]),
                .define("TESTING")
            ]
        )
    ]
)
