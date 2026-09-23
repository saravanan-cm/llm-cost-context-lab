import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// In dev, /api is proxied to the FastAPI backend so no CORS setup is needed.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".");
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": env.VITE_PROXY_TARGET || "http://localhost:8000",
      },
    },
  };
});
