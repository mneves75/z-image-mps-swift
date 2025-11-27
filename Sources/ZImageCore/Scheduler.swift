import Foundation
import MLX

/// FlowMatch Euler Discrete scheduler (matches diffusers FlowMatchEulerDiscreteScheduler).
/// Computes sigma schedule and provides a single-step Euler update.
public struct FlowMatchEulerDiscreteScheduler {
    public let numTrainTimesteps: Int
    public let shift: Float
    public let timesteps: [Float]
    public let sigmas: [Float]  // includes trailing zero

    public init(numTrainTimesteps: Int = 1000, shift: Float = 3.0, inferenceSteps: Int) {
        precondition(inferenceSteps > 0, "inferenceSteps must be positive")
        self.numTrainTimesteps = numTrainTimesteps
        self.shift = shift

        // Helpers mimic diffusers private methods
        func sigmaToT(_ sigma: Float) -> Float {
            sigma * Float(numTrainTimesteps)
        }

        // Compute base sigma range from training schedule
        let sigmaMax: Float = shift * 1.0 / (1.0 + (shift - 1.0) * 1.0) // ==1.0 when shift>=1
        let minSigmaUnshifted: Float = 1.0 / Float(numTrainTimesteps)
        let sigmaMin: Float = shift * minSigmaUnshifted / (1.0 + (shift - 1.0) * minSigmaUnshifted)

        let tStart = sigmaToT(sigmaMax)
        let tEnd = sigmaToT(sigmaMin)

        // linspace inclusive endpoints, count = inferenceSteps
        let ts: [Float] = (0..<inferenceSteps).map { i in
            if inferenceSteps == 1 { return tEnd }
            let a = Float(i)
            let b = Float(inferenceSteps - 1)
            return tStart + (tEnd - tStart) * (a / b)
        }

        // sigmas = shift * (t / num_train) / (1 + (shift-1)*(t/num_train))
        let baseSigmas = ts.map { t -> Float in
            let s = t / Float(numTrainTimesteps)
            return shift * s / (1.0 + (shift - 1.0) * s)
        }

        self.timesteps = baseSigmas.map { $0 * Float(numTrainTimesteps) }
        self.sigmas = baseSigmas + [0.0]
    }

    /// Single Euler step in flow-matching space.
    /// - Parameters:
    ///   - modelOutput: predicted velocity/noise (MLXArray)
    ///   - sample: current latent (MLXArray)
    ///   - sigma: current sigma value
    ///   - nextSigma: next sigma value
    /// - Returns: updated latent
    public func step(modelOutput: MLXArray, sample: MLXArray, sigma: Float, nextSigma: Float) -> MLXArray {
        let dt = nextSigma - sigma
        let drift = modelOutput * dt
        return sample + drift
    }
}
