import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite 配置：前端开发服务器端口 3000，并将 /api 请求代理到后端 8000
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
