"""Application entry point for the Vehicle Classification API."""

import uvicorn


def main():
    """Start the FastAPI application server."""
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()