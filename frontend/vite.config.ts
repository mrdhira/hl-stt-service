import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The backend is a separate uvicorn process in dev; in production Caddy serves
// the built assets and the API from the same origin, so the app always talks to
// relative paths and only the dev server needs this proxy.
const BACKEND = process.env.VITE_DEV_BACKEND ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/health': BACKEND,
      '/models': BACKEND,
      '/transcribe': BACKEND,
      '/runs': BACKEND,
      '/stream': { target: BACKEND, ws: true },
    },
  },
  build: {
    // Homelab box, no CDN: one small chunk beats many round trips.
    chunkSizeWarningLimit: 700,
  },
})
