// swift-tools-version:6.2
import PackageDescription

let package = Package(
  name: "zunder",
  platforms: [.macOS(.v14)],
  products: [.library(name: "ZunderCore", targets: ["ZunderCore"]), .executable(name: "zunder", targets: ["zunder"])],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", exact: "3.32.3"),
    .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
  ],
  targets: [
    .target(name: "ZunderCore", dependencies: [
      .product(name: "MLXLLM", package: "mlx-swift-lm"),
      .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
      .product(name: "MLXGuidedGeneration", package: "mlx-swift-lm"),
      .product(name: "Tokenizers", package: "swift-transformers"),
    ]),
    .executableTarget(name: "zunder", dependencies: ["ZunderCore"]),
    .testTarget(name: "ZunderCoreTests", dependencies: ["ZunderCore"]),
  ],
  swiftLanguageModes: [.v5]
)
