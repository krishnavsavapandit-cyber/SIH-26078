/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        meteo: {
          900: '#070d19',
          850: '#0c1527',
          800: '#111f38',
          700: '#1c2e4f',
          600: '#2b4474',
          accent: '#00d2ff',
          alert: '#ff3366',
          warning: '#ffb300',
          success: '#00e676',
        }
      },
      fontFamily: {
        mono: ['Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
