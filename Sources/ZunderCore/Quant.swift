import Foundation
import MLXLMCommon

struct TensorInfo: Equatable { var dtype: String; var shape: [Int] }

/// safetensors header only: 8-byte LE length + JSON.
func safetensorHeader(_ url: URL) throws -> [String: TensorInfo] {
  let f = try FileHandle(forReadingFrom: url); defer { try? f.close() }
  guard let n = try f.read(upToCount: 8), n.count == 8 else { throw ZError("bad safetensors: \(url.lastPathComponent)") }
  let len = n.withUnsafeBytes { Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))) }
  guard len < 1 << 28, let h = try f.read(upToCount: len), let j = try JSONSerialization.jsonObject(with: h) as? JSON else { throw ZError("bad safetensors header: \(url.lastPathComponent)") }
  var out: [String: TensorInfo] = [:]
  for (k, v) in j { if let d = v as? JSON, let dt = d["dtype"] as? String, let sh = d["shape"] as? [Int] { out[k] = TensorInfo(dtype: dt, shape: sh) } }
  return out
}

/// Bits per quantised path, inferred like mlx-lm's infer_quant_config: bits = 32 * packed_cols / (scales_cols * group_size).
/// Returns a per-layer config only if it differs from `base` (flint-int4: global 4 bit, embeddings 8 bit, no per-layer entries in config.json).
func inferQuantization(_ headers: [String: TensorInfo], base: BaseConfiguration.PerLayerQuantization) -> BaseConfiguration.PerLayerQuantization? {
  guard let g = base.quantization, g.mode == .affine else { return nil }
  var per = base.perLayerQuantization, changed = false
  let bytes = ["U8": 1, "I8": 1, "U32": 4, "I32": 4, "F32": 4]
  for (k, w) in headers where k.hasSuffix(".weight") {
    let prefix = String(k.dropLast(7))
    guard let s = headers[prefix + ".scales"], let wc = w.shape.last, let sc = s.shape.last, let b = bytes[w.dtype], sc > 0 else { continue }
    let path = prefix.replacingOccurrences(of: #"\.experts\.\d+\."#, with: ".switch_mlp.", options: .regularExpression)
    let gs = base.quantization(layer: path)?.groupSize ?? g.groupSize   // per-module group size (gemma mixed_4_6: global 32, modules 64)
    let packed = wc * b / 4, inDim = sc * gs
    guard packed > 0, 32 * packed % inDim == 0 else { continue }
    let bits = 32 * packed / inDim
    guard [2, 3, 4, 5, 6, 8].contains(bits) else { continue }
    // per-expert files are stacked into switch_mlp by sanitize
    if base.quantization(layer: path)?.bits != bits { per[path] = .quantize(.init(groupSize: gs, bits: bits)); changed = true }
  }
  return changed ? .init(quantization: g, perLayerQuantization: per) : nil
}

func inferQuantization(dir: URL, base: BaseConfiguration.PerLayerQuantization) throws -> BaseConfiguration.PerLayerQuantization? {
  var all: [String: TensorInfo] = [:]
  for u in try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) where u.pathExtension == "safetensors" {
    all.merge(try safetensorHeader(u)) { a, _ in a }
  }
  return inferQuantization(all, base: base)
}

struct NoProcessor: UserInputProcessor {
  func prepare(input: UserInput) async throws -> LMInput { throw ZError("text generation goes through Engine") }
}
