import type { ReactNode } from 'react'

export interface MetricProps {
  label: string
  value: ReactNode
  hint?: string
  tone?: 'good' | 'warn' | 'bad' | 'none'
}

export function Metric({ label, value, hint, tone = 'none' }: MetricProps) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className={`value tone-${tone}`}>{value}</div>
      {hint && <div className="hint">{hint}</div>}
    </div>
  )
}

export function MetricGrid({ items }: { items: MetricProps[] }) {
  return (
    <div className="metrics">
      {items.map((item) => (
        <Metric key={item.label} {...item} />
      ))}
    </div>
  )
}
