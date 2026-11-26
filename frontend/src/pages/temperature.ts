/** softmax(log p / T): the same temperature scaling the backend applies before its review rule. */
export function temperatureScale(probs: number[], t: number): number[] {
  const logits = probs.map(p => Math.log(Math.min(1, Math.max(p, 1e-12))) / t)
  const top = Math.max(...logits)
  const exp = logits.map(l => Math.exp(l - top))
  const sum = exp.reduce((a, b) => a + b, 0)
  return exp.map(v => v / sum)
}
export function topTwo(probs: number[]) {
  const order = probs.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0])
  return { first: order[0][1], second: order[1][1], margin: order[0][0] - order[1][0], top: order[0][0] }
}
