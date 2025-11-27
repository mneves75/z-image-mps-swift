import Foundation

/// GPT-2 style Byte Pair Encoding tokenizer compatible with Qwen2 tokenizer files.
/// Implements byte-level pretokenization, BPE merges, and vocab lookup.
public struct QwenTokenizer {
    private let vocab: [String: Int]
    private let mergesRanks: [String: Int]
    private let byteEncoder: [UInt8: String]
    private let byteDecoder: [String: UInt8]
    public init(vocabURL: URL, mergesURL: URL) throws {
        let vocabData = try Data(contentsOf: vocabURL)
        let vocabJSON = try JSONSerialization.jsonObject(with: vocabData) as! [String: Int]
        self.vocab = vocabJSON

        // merges file: first line is header
        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        var ranks: [String: Int] = [:]
        let lines = mergesText.split(separator: "\n")
        for (i, line) in lines.enumerated() {
            if i == 0 { continue } // skip header
            let parts = line.split(separator: " ")
            if parts.count != 2 { continue }
            let key = "\(parts[0]) \(parts[1])"
            ranks[key] = i - 1
        }
        self.mergesRanks = ranks

        let (enc, dec) = Self.buildByteEncoder()
        self.byteEncoder = enc
        self.byteDecoder = dec
    }

    public func encode(_ text: String) -> [Int] {
        let tokens = tokenize(text)
        return tokens.compactMap { vocab[$0] }
    }

    // MARK: - Internal helpers

    private func tokenize(_ text: String) -> [String] {
        // Byte-level BPE: treat entire text as bytes.
        return bpe(text)
    }

    private func bpe(_ token: String) -> [String] {
        let bytes = Array(token.utf8)
        var word = bytes.map { byteEncoder[$0]! }
        if word.count == 1 {
            return [word.joined()]
        }
        var pairs = getPairs(word)
        while true {
            guard let bigram = pairs.min(by: { (a, b) -> Bool in
                let ra = mergesRanks[a] ?? Int.max
                let rb = mergesRanks[b] ?? Int.max
                return ra < rb
            }) else { break }

            guard mergesRanks[bigram] != nil else { break }

            let parts = bigram.split(separator: " ").map(String.init)
            let first = parts[0], second = parts[1]
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == first && word[i + 1] == second {
                    newWord.append(first + second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
            if word.count == 1 { break }
            pairs = getPairs(word)
        }
        return word
    }

    private func getPairs(_ symbols: [String]) -> Set<String> {
        var pairs = Set<String>()
        for i in 0..<(symbols.count - 1) {
            pairs.insert(symbols[i] + " " + symbols[i + 1])
        }
        return pairs
    }

    // Byte encoder/decoder from GPT-2 reference.
    private static func buildByteEncoder() -> ([UInt8: String], [String: UInt8]) {
        var bs: [UInt8] = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs.map { Int($0) }
        var n = 0
        for b in 0...255 {
            if !bs.contains(UInt8(b)) {
                bs.append(UInt8(b))
                cs.append(256 + n)
                n += 1
            }
        }
        let encoder: [UInt8: String] = Dictionary(uniqueKeysWithValues: zip(bs, cs.map { String(UnicodeScalar($0)!) }))
        let decoder: [String: UInt8] = Dictionary(uniqueKeysWithValues: encoder.map { ($0.value, $0.key) })
        return (encoder, decoder)
    }
}
