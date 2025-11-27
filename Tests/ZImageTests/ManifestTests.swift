import Foundation
import Testing
@testable import ZImageCore

struct ManifestTests {
    @Test
    func buildsManifestAndHashes() throws {
        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        let file = tempDir.appendingPathComponent("a.txt")
        try "hello".data(using: .utf8)!.write(to: file)

        let manifest = try ManifestBuilder.build(for: tempDir, relativeTo: tempDir)
        #expect(manifest.files.count == 1)
        #expect(manifest.files.first?.path == "a.txt")
        #expect(manifest.files.first?.size == 5)
        #expect(manifest.files.first?.sha256.count == 64)
    }
}
