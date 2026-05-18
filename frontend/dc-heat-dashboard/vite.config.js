import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8765", changeOrigin: true },
      "/outputs": { target: "http://127.0.0.1:8765", changeOrigin: true },
      "/opt-out": { target: "http://127.0.0.1:8765", changeOrigin: true },
    },
  },
});