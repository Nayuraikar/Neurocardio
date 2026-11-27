import { formatValue, type Fmt } from './format'

/** A displayed statistic with its source ("<artifact>#<path>"), so the checker can verify it. */
export function Num({ v, src, fmt = 'int' }: { v: unknown; src: string; fmt?: Fmt }) {
  return <span className="num" data-source={src} data-fmt={fmt}>{formatValue(v, fmt)}</span>
}
