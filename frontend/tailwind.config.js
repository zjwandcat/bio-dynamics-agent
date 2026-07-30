/** @type {import('tailwindcss').Config} */
// Tailwind 配置：扫描入口与组件文件，扩展深色主题调色板
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // 深色主题背景层级
        slate: {
          900: '#0f172a',
          800: '#1e293b',
          700: '#334155',
        },
        // 强调色
        indigo: {
          500: '#6366f1',
        },
        emerald: {
          500: '#10b981',
        },
        rose: {
          500: '#f43f5e',
        },
      },
    },
  },
  plugins: [],
}
