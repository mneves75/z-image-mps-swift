# Code Implementation Task Spec — MLX-Swift CLI for Z-Image/Qwen
Version: 2025-11-27 • Target: macOS 15+ (Apple Silicon), Xcode 26.x, Swift 6.2, MLX-Swift 0.20+

## 1) Problem Summary / Scope / Non-Goals
- **Problem Summary:** Rebuild the Z-Image Turbo text-to-image pipeline as a Swift CLI using MLX-Swift, matching Python parity (tokenizer, Qwen3 text encoder, DiT, VAE, scheduler) and supporting local generation with cached Hugging Face weights (primary: `Tongyi-MAI/Z-Image-Turbo`; optional: `Qwen/Qwen3-32B-MLX-bf16`). Provide tooling to fetch/convert/verify weights, run parity tests, and expose a robust CLI without GUI. Must follow Carmack-level correctness and modern Swift/MLX best practices. Reuse proven patterns from `mzbac/qwen.image.swift` where they improve correctness/performance.
- **Scope:** macOS CLI binary (`z-image-cli`), core library (`ZImageCore`), tools (`z-image-tools`) for HF download/convert/manifest, automated tests/fixtures, docs. Include system-requirement checks, HF CLI integration, parity validation for tokenizer/encoder/scheduler/vae/dit. Support CPU fallback only when explicitly requested.
- **Non-Goals:** No GUI, no cloud backends, no Windows/Linux ports, no training/fine-tuning, no model editing beyond dtype conversion.
- **Open Questions:** (a) Do we need mixed-precision knobs per stage? (b) Preferred default cache root (XDG vs HOME)? (c) Ship tiny test weights in repo or rely on live HF fetch during CI?

## 2) System Context
- Architecture: SwiftPM workspace with three targets (`ZImageCore` library, `ZImageCLI` executable, `ZImageTools` utility). MLX-Swift provides tensor + Metal/CPU backends. Uses safetensors via MLX loaders.
- Runtime: macOS 15+, Apple Silicon (M3/M4). Swift 6.2, Xcode 26.1+, MLX-Swift >=0.20. Metal available by default; CPU fallback optional. Python venv only for fixture generation.
- External deps: `huggingface-cli` (for auth + downloads), HF models (`Tongyi-MAI/Z-Image-Turbo`, optional `Qwen/Qwen3-32B-MLX-bf16`), `safetensors` for conversion. No network during inference.
- Constraints: Peak mem <10 GB @1024², startup <5s after cache warm. Determinism per seed. Fail fast on missing/invalid weights. Parity against HF reference for tokenizer + Qwen encoder (stats fixtures). DType policy: bf16 weights, fp32 VAE decode.

## 3) Functional Requirements
- **FR-1 CLI Inference**: Provide command `z-image-cli` with args: prompt, negative prompt, steps (1–50), guidance (0–20), aspect preset or `--height/--width` (multiples of 64, 256–2048), num-images (1–16), seed (int, auto-increment), device (`auto|metal|cpu`, cpu requires `--allow-cpu`), output dir. Validates inputs, prints run summary (device, dtype, steps, size, seed, timings), exits non-zero on typed errors.
- **FR-2 Weight Management**: `z-image-tools fetch` downloads required HF artifacts via `huggingface-cli` with optional token; `convert` casts safetensors to unified dtype and emits `manifest.json` (sha256, shapes, dtype); `verify` checks manifest hashes; `clean` removes cache. Paths default `~/.cache/z-image-mlx/` (override `XDG_CACHE_HOME`).
- **FR-3 Text Stack**: Load tokenizer (vocab/merges), encode text ids, run Qwen3 text encoder with RoPE + causal mask matching HF, output float32 embeddings. Parity tests compare mean/std/slice to fixtures. Supports sharded safetensors index.
- **FR-4 Diffusion Core**: Load scheduler (DPMSolver++ ancestral), DiT, VAE; run denoise loop with seed determinism; VAE decode in fp32; output PNG files with deterministic naming `z-image-YYYYMMDD-HHMMSS-{i}.png`.
- **FR-5 Logging & Telemetry**: Structured OSLog categories (`pipeline`, `weights`, `io`); timings per stage (load/encode/denoise/decode/save). Optional `--log-prompts` to include prompts; default avoids prompt logging.
- **FR-6 Error Model**: Typed errors (`WeightsMissing`, `IntegrityMismatch`, `InvalidDimension`, `UnsupportedDevice`, `ConversionFailed`, `PipelineError`). Human-friendly messages, non-zero exit codes.

Edge cases per FRs: invalid dims, missing manifest/weights, gated HF repo without token, Metal unavailable, CPU selected without flag, cache corruption (hash mismatch), partial shard downloads, tokenizer file mismatch, scheduler step count 0, negative guidance, output dir not writable.

## 4) Data Model & Interfaces
- **Entities**:
  - `Manifest`: {version, files:[{path, sha256, dtype, shape}]}
  - `TextEncoderConfig`: {hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, intermediate_size, head_dim, rms_norm_eps, rope_theta=1e6 (assumed)}
  - `PipelineConfig`: combines transformer/vae config, dtype policy, cache paths.
  - `Weights` store: safetensors shards, converted `.safetensors` (bf16), manifests, tokenizer files, scheduler JSON.
- **APIs/Functions** (Swift):
  - `WeightLoader.loadShardedSafetensors(dir:indexFile) -> [String: MLXArray]`
  - `WeightsRepository.verify(url) -> Manifest`
  - `Qwen3Encoder.init(weights:[String:MLXArray], config:TextEncoderConfig)`; `encode(ids:[Int], stream:StreamOrDevice)`
  - `Scheduler.step(sample, sigma, ... ) -> sample`
  - CLI entry `run()` orchestrates: validate args -> load weights -> tokenize -> encode -> denoise -> decode -> save -> log summary.

## 5) Task Decomposition For Agents
- **T-1 Environment Gate**: Implement `make doctor` (or Swift tool) to assert Xcode 26+, Swift 6.2, Metal available, MLX device listing, HF CLI present; fail with actionable hints.
- **T-2 HF Fetch Tooling**: Add `z-image-tools fetch --model Tongyi-MAI/Z-Image-Turbo --target <dir> [--token]` using `huggingface-cli` (shell). Support optional `--model Qwen/Qwen3-32B-MLX-bf16`.
- **T-3 Manifest & Convert**: Implement `convert` to cast safetensors to bf16 (VAE decode kept fp32), emit `manifest.json`. Add hashing helper. Tests with tiny fixture tensors.
- **T-4 Tokenizer Loader**: Swift loader for vocab/merges/tokenizer_config; parity test on fixture ids (already in repo).
- **T-5 Qwen3 Encoder Parity**: Finish RoPE + causal mask + repeat_kv to match HF stats; expose `encodeLayer` for fixture checks; pass mean/std/slice fixtures. Add stream plumbing and avoid Metal teardown crash.
- **T-6 Scheduler**: Ensure DPMSolver++ (ancestral) parameters match Python reference; test sigmas against fixture.
- **T-7 Pipeline Wiring**: Compose tokenizer -> encoder -> denoiser -> vae -> PNG saver; support multi-image seeds; dtype/device selection; fp32 VAE decode.
- **T-8 CLI UX**: ArgumentParser with all flags, presets, validation, allow-cpu guard, run summary printing.
- **T-9 Logging & Errors**: OSLog categories, typed errors, graceful exit codes; optional prompt logging flag.
- **T-10 Docs**: Update README quickstart, weight instructions, troubleshooting; add updated engineering spec (this file).
- **T-11 CI & Tests**: Add Swift test suite to GH Actions/macOS runner; ensure `swift test` runs offline with fixtures; include smoke test with tiny weights if feasible.

Dependencies: T-1 before others; T-2 before T-3; T-4/5 before T-7; T-6 before T-7; T-7 before T-8/9; Tests/Docs after features. Risks: Qwen parity (numerical), Metal vs CPU divergence, HF download speed.

## 6) Testing & Verification
- **Unit**: tokenizer ids fixture; manifest hash verification; dimension validator; seed sequence; file naming; device selection guard.
- **Encoder Parity**: Compare mean/std/slice for full encoder and layer0 vs fixtures (zimage_model_fixtures.json, text_layer0.json) on CPU stream; tolerance 5e-3/5e-2.
- **Scheduler**: sigmas array equality to fixture.
- **Conversion**: round-trip load/convert/save on tiny tensors; manifest hashes stable.
- **Pipeline Smoke**: mocked weights -> ensure shapes, no crash; real weights (if available) single step 64×64 deterministic stats snapshot.
- **Non-functional**: memory cap check (<10 GB for 1024², if measurable), startup timing (<5s) manual.

## 7) Review Checklist
- Requirements mapped to tasks; no TODOs left.
- All public APIs typed with shapes/dtypes; examples provided.
- Error types returned with non-zero exits; messages human-friendly.
- Tests cover happy path + edge cases listed (dims, missing weights, hash mismatch, CPU guard, parity fixtures).
- HF tooling uses `huggingface-cli` and caches to user path; no network during inference.
- Logging enabled and optional prompt logging off by default.
- Docs updated (README, this spec) with prerequisites and commands.

