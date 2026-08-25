import { expect, test } from "@playwright/test";
import { createServer, type Server } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";

let server: Server;

test.beforeAll(async () => {
  server = createServer(async (request, response) => {
    const pathname = new URL(request.url ?? "/", "http://127.0.0.1").pathname;
    const file = pathname.endsWith("/") ? "index.html" : pathname.slice(1);
    const resolved = join(process.cwd(), "apps/playground/dist", file);
    try {
      const body = await readFile(resolved);
      response.setHeader("content-type", contentType(resolved));
      response.end(body);
    } catch {
      const body = await readFile(join(process.cwd(), "apps/playground/dist/index.html"));
      response.setHeader("content-type", "text/html");
      response.end(body);
    }
  });
  await new Promise<void>((resolve) => server.listen(5174, "127.0.0.1", resolve));
});

test.afterAll(async () => {
  await new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
});

test("renders Gradlith playground and benchmark route", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Gradlith" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Start training" })).toBeVisible();

  await page.goto("/benchmarks/");
  await expect(page.getByRole("heading", { name: "Kernel Bench" })).toBeVisible();
});

function contentType(path: string): string {
  if (extname(path) === ".js") {
    return "text/javascript";
  }
  if (extname(path) === ".css") {
    return "text/css";
  }
  return "text/html";
}
