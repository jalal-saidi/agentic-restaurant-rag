"""Run the API with ``python -m connoisseur.api``."""

from __future__ import annotations

import os


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install the API dependencies before starting the server") from exc
    uvicorn.run(
        "connoisseur.api.app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
