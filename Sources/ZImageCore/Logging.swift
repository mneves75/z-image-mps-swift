import Foundation
import OSLog

public enum Log {
    private static let subsystem = "dev.zimage.mlx"

    private static let pipelineLogger = Logger(subsystem: subsystem, category: "pipeline")
    private static let weightsLogger = Logger(subsystem: subsystem, category: "weights")
    private static let cliLogger = Logger(subsystem: subsystem, category: "cli")

    public static func pipeline(_ message: String) {
        pipelineLogger.info("\(message, privacy: .public)")
    }

    public static func weights(_ message: String) {
        weightsLogger.info("\(message, privacy: .public)")
    }

    public static func cli(_ message: String) {
        cliLogger.info("\(message, privacy: .public)")
    }

    public static func error(_ message: String) {
        cliLogger.error("\(message, privacy: .public)")
    }
}
