import Foundation

public enum OutputNamer {
    /// Build a deterministic filename with timestamp and optional index suffix.
    public static func filename(index: Int = 0,
                                timestamp: Date = .init(),
                                base: String = "z-image") -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        let stamp = formatter.string(from: timestamp)
        let suffix = index > 0 ? "-\(index + 1)" : ""
        return "\(base)-\(stamp)\(suffix).png"
    }
}

public enum PathResolver {
    public static func resolveOutput(output: String?, outdir: String?, index: Int, timestamp: Date = .init()) throws -> URL {
        let fm = FileManager.default
        if let output {
            let expanded = NSString(string: output).expandingTildeInPath
            var isDir: ObjCBool = false
            if fm.fileExists(atPath: expanded, isDirectory: &isDir), isDir.boolValue || expanded.hasSuffix("/") {
                let dirURL = URL(fileURLWithPath: expanded, isDirectory: true)
                return dirURL.appendingPathComponent(OutputNamer.filename(index: index, timestamp: timestamp))
            }
            let url = URL(fileURLWithPath: expanded)
            let dir = url.deletingLastPathComponent()
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            if index > 0 {
                let ext = url.pathExtension.isEmpty ? "png" : url.pathExtension
                let basename = url.deletingPathExtension().lastPathComponent
                let renamed = dir.appendingPathComponent("\(basename)-\(index + 1).\(ext)")
                return renamed
            }
            return url
        }

        let baseDirPath = outdir ?? "output"
        let baseDir = URL(fileURLWithPath: NSString(string: baseDirPath).expandingTildeInPath, isDirectory: true)
        try fm.createDirectory(at: baseDir, withIntermediateDirectories: true)
        return baseDir.appendingPathComponent(OutputNamer.filename(index: index, timestamp: timestamp))
    }
}
