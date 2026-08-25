import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  base: process.env.GITHUB_PAGES === "true" ? "/Gradlith/" : "/",
  plugins: [react()],
  worker: {
    format: "es"
  },
  resolve: {
    alias: {
      "@gradlith/core": fileURLToPath(new URL("../../packages/core/src/index.ts", import.meta.url))
    }
  },
  server: {
    port: 5173
  }
});
