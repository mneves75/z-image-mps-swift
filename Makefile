SWIFT_BUILD = xcodebuild -scheme z-image-cli

.PHONY: build test fetch-weights convert-weights format

build:
	swift build -c release

test:
	swift test

fetch-weights:
	@echo "TODO: Implement weight fetch (Hugging Face). Use: z-image-tools fetch --output <dir>"

convert-weights:
	@echo "TODO: Implement weight conversion to MLX. Use: z-image-tools convert --source <dir> --output <dir>"

format:
	swift format --in-place --recursive Sources Tests || true
