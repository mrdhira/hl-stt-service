import { useCallback, useEffect, useState } from 'react'

import { ApiError, getModels, getRuns } from './api/client'
import { RUNS_LIMIT } from './api/config'
import type { ModelInfo, ModelsResponse, RunRow } from './api/types'
import { Reports } from './tabs/Reports'
import { RecordSend } from './tabs/RecordSend'
import { StreamTab } from './tabs/StreamTab'

const TABS = [
  { id: 'record', label: 'Record & Send' },
  { id: 'stream', label: 'Stream' },
  { id: 'reports', label: 'Reports' },
] as const

type TabId = (typeof TABS)[number]['id']

export default function App() {
  const [tab, setTab] = useState<TabId>('record')

  const [modelsInfo, setModelsInfo] = useState<ModelsResponse | null>(null)
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [model, setModel] = useState('')

  const [runs, setRuns] = useState<RunRow[]>([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [runsError, setRunsError] = useState<string | null>(null)
  const [limit, setLimit] = useState(RUNS_LIMIT)

  const loadModels = useCallback(async (rescan = false) => {
    try {
      const response = await getModels(rescan)
      setModelsInfo(response)
      setModelsError(null)
      // Default to the first usable model, but never override a live choice.
      setModel((current) => {
        if (current && response.models.some((m) => m.key === current)) return current
        return (
          response.models.find((m) => m.available)?.key ?? response.models[0]?.key ?? ''
        )
      })
    } catch (cause) {
      setModelsError(cause instanceof ApiError ? cause.message : String(cause))
    }
  }, [])

  const loadRuns = useCallback(async (rowLimit: number) => {
    setRunsLoading(true)
    try {
      const response = await getRuns({ limit: rowLimit })
      setRuns(response.runs)
      setRunsError(null)
    } catch (cause) {
      setRunsError(cause instanceof ApiError ? cause.message : String(cause))
    } finally {
      setRunsLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadModels()
  }, [loadModels])

  useEffect(() => {
    void loadRuns(limit)
  }, [loadRuns, limit])

  const models: ModelInfo[] = modelsInfo?.models ?? []
  const onRunSaved = useCallback(() => {
    void loadRuns(limit)
    // A model is only marked `loaded` after its first decode; refresh the badge.
    void loadModels()
  }, [loadRuns, loadModels, limit])

  const anyAvailable = models.some((m) => m.available)

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>hl-stt · benchmark harness</h1>
          <div className="sub">
            Offline speech-to-text POC — English + Japanese, CPU only
          </div>
        </div>
        <div className="row" style={{ gap: 8 }}>
          {modelsInfo && (
            <>
              <span className={`badge ${modelsInfo.sherpa_onnx_available ? 'ok' : 'off'}`}>
                <span className="dot" />
                sherpa-onnx {modelsInfo.sherpa_onnx_version ?? 'missing'}
              </span>
              <span className="badge off">{modelsInfo.num_threads} threads</span>
            </>
          )}
          <button className="btn" onClick={() => void loadModels(true)}>
            Rescan models
          </button>
        </div>
      </header>

      <nav className="tabs" role="tablist">
        {TABS.map((entry) => (
          <button
            key={entry.id}
            role="tab"
            aria-selected={tab === entry.id}
            onClick={() => setTab(entry.id)}
          >
            {entry.label}
          </button>
        ))}
      </nav>

      {modelsError && (
        <div className="notice error" style={{ marginBottom: 16 }}>
          {modelsError}
        </div>
      )}

      {modelsInfo && !anyAvailable && (
        <div className="notice warn" style={{ marginBottom: 16 }}>
          No models are available. Extract the sherpa-onnx model directories under{' '}
          <span className="mono">{modelsInfo.models_dir}</span> and hit “Rescan models”.
          Reports still works — it reads whatever is already in the database.
        </div>
      )}

      {tab === 'record' && (
        <RecordSend
          models={models}
          model={model}
          onModelChange={setModel}
          onRunSaved={onRunSaved}
        />
      )}

      {tab === 'stream' && (
        <StreamTab
          models={models}
          model={model}
          onModelChange={setModel}
          onRunSaved={onRunSaved}
        />
      )}

      {tab === 'reports' && (
        <Reports
          models={models}
          runs={runs}
          loading={runsLoading}
          error={runsError}
          limit={limit}
          onLimitChange={setLimit}
          onRefresh={() => void loadRuns(limit)}
        />
      )}
    </div>
  )
}
