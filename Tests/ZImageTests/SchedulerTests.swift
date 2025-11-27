import Foundation
import Testing
import MLX
@testable import ZImageCore

struct SchedulerTests {
    @Test
    func sigmaMatchesFixture() throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: "Tests/fixtures/zimage_fixtures.json"))
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let sigmas = json["sigmas_8"] as! [Double]

        let scheduler = FlowMatchEulerDiscreteScheduler(inferenceSteps: 8)
        let computed = scheduler.sigmas.map { Double($0) }
        #expect(computed.count == sigmas.count)
        for (a, b) in zip(computed, sigmas) {
            #expect(abs(a - b) < 1e-5)
        }
    }
}
