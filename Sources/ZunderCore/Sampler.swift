import MLX

public struct SampleParams: Sendable {
  public var temp: Float = 1.0, topP: Float = 0.95
  public var topK = 20
  public var repPen: Float = 1, repCtx = 20   // mlx-lm repetition_penalty / repetition_context_size; 1 = off
  public init(temp: Float = 1.0, topP: Float = 0.95, topK: Int = 20) { self.temp = temp; self.topP = topP; self.topK = topK }
}

/// mlx-lm make_repetition_penalty: logits of the context tokens are divided by `pen` if >= 0, else multiplied.
public func repPenalty(_ l: MLXArray, _ toks: MLXArray, _ pen: Float) -> MLXArray {
  let s = takeAlong(l, toks, axis: -1)
  return putAlong(l, toks, values: which(s .< 0, s * pen, s / pen), axis: -1)
}

/// logits [V] (masked entries -inf) -> token id scalar. top-k via partition (k small), top-p on the k survivors.
public func sample(_ logits: MLXArray, _ p: SampleParams) -> MLXArray {
  if p.temp <= 0 { return argMax(logits, axis: -1) }
  let n = logits.dim(-1)
  let l = logits / p.temp
  var idx: MLXArray, v: MLXArray
  if p.topK > 0 && p.topK < n {
    idx = argPartition(-l, kth: p.topK - 1, axis: -1)[..<p.topK]
    v = takeAlong(l, idx, axis: -1)
  } else { idx = MLXArray(0 ..< Int32(n)); v = l }
  let o = argSort(-v, axis: -1)
  idx = takeAlong(idx, o, axis: -1); v = takeAlong(v, o, axis: -1)
  if p.topP < 1 {
    let pr = softmax(v, axis: -1)
    v = which((cumsum(pr, axis: -1) - pr) .< p.topP, v, MLXArray(-Float.infinity))
  }
  return idx[categorical(v)]
}
