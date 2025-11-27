import Foundation
import CryptoKit

public enum FileHasher {
    public static func sha256(url: URL) throws -> String {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let digest = SHA256.hash(data: data)
        return digest.compactMap { String(format: "%02x", $0) }.joined()
    }
}

public struct FileRecord: Codable, Sendable, Equatable {
    public let path: String
    public let sha256: String
    public let size: Int
}

public struct Manifest: Codable, Sendable, Equatable {
    public let createdAt: Date
    public let files: [FileRecord]
}

public enum ManifestBuilder {
    public static func build(for directory: URL, relativeTo base: URL? = nil) throws -> Manifest {
        let fm = FileManager.default
        let dir = directory.resolvingSymlinksInPath()
        guard let enumerator = fm.enumerator(at: dir, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) else {
            throw ZImageError.weightsMissing(directory)
        }
        var records: [FileRecord] = []
        for case let fileURL as URL in enumerator {
            let normalizedFile = fileURL.resolvingSymlinksInPath()
            let resourceValues = try fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard resourceValues.isRegularFile == true else { continue }
            let hash = try FileHasher.sha256(url: normalizedFile)
            let size = resourceValues.fileSize ?? 0
            let relativePath: String
            if let base {
                let normBase = base.resolvingSymlinksInPath()
                let prefix = normBase.path.hasSuffix("/") ? normBase.path : normBase.path + "/"
                var path = normalizedFile.path
                if path.hasPrefix(prefix) {
                    path.removeFirst(prefix.count)
                }
                relativePath = path
            } else {
                relativePath = normalizedFile.lastPathComponent
            }
            records.append(FileRecord(path: relativePath, sha256: hash, size: size))
        }
        return Manifest(createdAt: Date(), files: records.sorted { $0.path < $1.path })
    }
}
