# models/

Model-agnostic intelligence layer. Nothing outside this folder should
import a specific provider SDK.

- `adapter.py` — `ModelAdapter` protocol every provider implements.
- `router.py` — picks an adapter per task (e.g. cheap model for
  classification, strong model for code generation).
- `inference.py` — retries, streaming, structured-output parsing.
- `providers/` — concrete adapters (Anthropic today; others later).
