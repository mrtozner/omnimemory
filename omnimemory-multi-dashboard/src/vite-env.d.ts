/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  readonly VITE_GATEWAY_URL: string
  readonly VITE_WS_METRICS_URL: string
  readonly VITE_WS_GATEWAY_URL: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
