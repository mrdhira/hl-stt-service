interface Props {
  text: string
  partial?: boolean
  placeholder?: string
}

export function Transcript({ text, partial, placeholder }: Props) {
  if (!text) {
    return (
      <div className="transcript">
        <span className="placeholder">{placeholder ?? 'Nothing yet.'}</span>
      </div>
    )
  }
  return <div className={`transcript${partial ? ' partial' : ''}`}>{text}</div>
}
