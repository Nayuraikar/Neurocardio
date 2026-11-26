import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const api = process.env.NEUROCARDIO_API ?? 'http://127.0.0.1:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': api } },
})
