import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    'process.env.WANDB_API_KEY': JSON.stringify(process.env.WANDB_API_KEY),
  },
  server: {
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
})
