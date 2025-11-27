import Foundation
import Testing
@testable import ZImageCore
@testable import ZImageCLI

struct ValidationTests {
    @Test
    func acceptsPresetDimensions() throws {
        try DimensionValidator.validate(width: 1024, height: 1024)
    }

    @Test
    func rejectsNonMultiple() {
        #expect(throws: ZImageError.self) {
            try DimensionValidator.validate(width: 1000, height: 1024)
        }
    }

    @Test
    func seedSequenceDeterministic() {
        let seeds = SeedGenerator.seeds(count: 3, baseSeed: 42)
        #expect(seeds == [42, 43, 44])
    }

    @Test
    func seedSequenceRandomHasCount() {
        let seeds = SeedGenerator.seeds(count: 2, baseSeed: nil)
        #expect(seeds.count == 2)
    }

    @Test
    func outputPathDirectoryExpands() throws {
        let url = try PathResolver.resolveOutput(output: nil, outdir: "~/tmp/zimg-test", index: 0, timestamp: Date(timeIntervalSince1970: 0))
        #expect(url.lastPathComponent.hasPrefix("z-image-19700101-000000"))
    }

    @Test
    func filenameIncrements() {
        let ts = Date(timeIntervalSince1970: 0)
        let name0 = OutputNamer.filename(index: 0, timestamp: ts)
        let name1 = OutputNamer.filename(index: 1, timestamp: ts)
        #expect(name0 != name1)
        #expect(name1.contains("-2"))
    }

    @Test
    func aspectLookup() {
        #expect(AspectPresets.preset(named: "16:9")?.width == 1280)
        #expect(AspectPresets.preset(named: "invalid") == nil)
    }

    @Test
    func deviceSelectionRespectsCPUFlag() {
        var threw = false
        do {
            _ = try DeviceSelector.select(preferred: .cpu, allowCPU: false)
        } catch ZImageError.cpuNotAllowed {
            threw = true
        } catch {
            threw = false
        }
        #expect(threw)
        let ok = try? DeviceSelector.select(preferred: .cpu, allowCPU: true)
        #expect(ok?.device == .cpu)
    }
}
