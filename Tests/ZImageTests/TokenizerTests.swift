import Foundation
import Testing
@testable import ZImageCore

struct TokenizerTests {
    @Test
    func loadsFixtureIds() throws {
        let tok = try QwenTokenizer(
            vocabURL: URL(fileURLWithPath: ".cache/hf/Z-Image-Turbo/tokenizer/vocab.json"),
            mergesURL: URL(fileURLWithPath: ".cache/hf/Z-Image-Turbo/tokenizer/merges.txt")
        )
        let ids = tok.encode("Analog film portrait of a skateboarder, shallow depth of field").prefix(13)
        let fixture = try Data(contentsOf: URL(fileURLWithPath: "Tests/fixtures/zimage_model_fixtures.json"))
        let json = try JSONSerialization.jsonObject(with: fixture) as! [String: Any]
        let expected = Array((json["token_ids_32"] as! [Int]).prefix(13))
        #expect(Array(ids) == expected)
    }
}
