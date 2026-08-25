# Contributing

Gradlith prioritizes correctness before visual polish.

Before submitting changes:

```bash
pnpm install
pnpm test
pnpm build
```

Guidelines:

- Do not add external machine-learning runtimes to the framework.
- Keep tensor math, autograd, training and optimizers implemented in Gradlith code.
- Add focused numerical tests for new operations.
- Prefer finite-difference gradient checks for differentiable behavior.
- Keep UI code in `apps/playground` and framework code in `packages/core`.

