import Foundation
import MLX
import MLXNN

public enum ActDtype: String { case fp16, bf16 }

/// `--dtype fp16|bf16|auto`, env ZUNDER_DTYPE. auto = fp16 where bf16 is emulated (Apple GPU g13/g14 = M1/M2 family), else bf16 (checkpoint dtype).
public func resolveDtype(_ s: String?) -> ActDtype {
  let v = (s ?? ProcessInfo.processInfo.environment["ZUNDER_DTYPE"] ?? "auto").lowercased()
  if v == "fp16" || v == "float16" { return .fp16 }
  if v == "bf16" || v == "bfloat16" { return .bf16 }
  let a = GPU.deviceInfo().architecture
  return a.hasPrefix("applegpu_g13") || a.hasPrefix("applegpu_g14") ? .fp16 : .bf16
}

/// Casts every floating parameter (norms, scales/biases, router, sinks, layer scalars) to one dtype; packed integer weights untouched.
/// All params must agree: bf16 activations x fp16 params promote to fp32 (mlx-swift #420).
public func castParameters(_ m: Module, to d: ActDtype) {
  let t: DType = d == .fp16 ? .float16 : .bfloat16
  let p = m.mapParameters { $0.dtype.isFloatingPoint && $0.dtype != t ? $0.asType(t) : $0 }
  m.update(parameters: p)
  eval(m)
}
