import Foundation
import Testing
@testable import ZImageCore

struct WeightsTests {
    @Test
    func manifestVerificationPasses() throws {
        let temp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        let file = temp.appendingPathComponent("w.bin")
        try Data([0, 1, 2, 3]).write(to: file)
        let hash = try FileHasher.sha256(url: file)
        let manifest = Manifest(createdAt: Date(), files: [FileRecord(path: "w.bin", sha256: hash, size: 4)])
        let manifestURL = temp.appendingPathComponent("manifest.json")
        let data = try JSONEncoder().encode(manifest)
        try data.write(to: manifestURL)

        let verified = try WeightsRepository.verify(at: temp)
        #expect(verified.files.count == 1)
    }

    @Test
    func manifestVerificationFailsOnMismatch() throws {
        let temp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        let file = temp.appendingPathComponent("w.bin")
        try Data([0, 1, 2, 3]).write(to: file)
        let manifest = Manifest(createdAt: Date(), files: [FileRecord(path: "w.bin", sha256: String(repeating: "0", count: 64), size: 4)])
        let manifestURL = temp.appendingPathComponent("manifest.json")
        let data = try JSONEncoder().encode(manifest)
        try data.write(to: manifestURL)

        var mismatch = false
        do {
            _ = try WeightsRepository.verify(at: temp)
        } catch ZImageError.integrityMismatch {
            mismatch = true
        }
        #expect(mismatch)
    }
}

struct WeightComponentTests {
    @Test
    func failsWhenComponentMissing() throws {
        let temp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        let manifest = Manifest(createdAt: Date(), files: [])
        let manifestURL = temp.appendingPathComponent("manifest.json")
        try JSONEncoder().encode(manifest).write(to: manifestURL)

        var missing = false
        do {
            try WeightsRepository.requireComponents(in: temp)
        } catch ZImageError.weightsMissing {
            missing = true
        }
        #expect(missing)
    }
}
