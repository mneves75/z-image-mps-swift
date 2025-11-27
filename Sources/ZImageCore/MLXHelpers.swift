import Foundation
import MLX

public enum MLXHelpers {
    /// Create 1-D arange [0, count) as float32
    public static func arange(_ count: Int, dtype: MLX.DType = .float32, stream: StreamOrDevice = .default) -> MLXArray {
        let values = (0..<count).map { Int32($0) }
        let base = MLXArray(values)
        return base.asType(dtype)
    }

    /// Concatenate arrays along axis.
    public static func concat(_ arrays: [MLXArray], axis: Int = 0) -> MLXArray {
        concatenated(arrays, axis: axis)
    }

    /// Repeat array along axis by count times using the provided stream/device.
    public static func repeatArray(_ array: MLXArray, count: Int, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray.repeated(array, count: count, axis: axis, stream: stream)
    }
}
