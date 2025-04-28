import argparse
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .middleware.logging import RequestResponseLoggingMiddleware
from .routers import api_router

app = FastAPI(title="MLX Omni Server")

# Add request/response logging middleware with custom levels
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # or ["http://localhost:8080"] if you host your page there
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


def build_parser():
    """Create and configure the argument parser for the server."""
    parser = argparse.ArgumentParser(description="MLX Omni Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to, defaults to 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10240,
        help="Port to bind the server to, defaults to 10240",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers to use, defaults to 1",
    )
    parser.add_argument(
        "--prompt-cache-capacity",
        type=int,
        default=8,
        help="Number of per-conversation prompt caches to keep in memory (default: 8)",
    )
    parser.add_argument(
        "--prompt-cache-max-tokens",
        type=int,
        default=None,
        help="Max tokens to retain in each prompt cache (default: no limit)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level, defaults to info",
    )
    return parser


def start():
    """Start the MLX Omni Server."""
    parser = build_parser()
    args = parser.parse_args()

    # Set log level through environment variable
    os.environ["MLX_OMNI_LOG_LEVEL"] = args.log_level
    # Prompt-cache tuning
    os.environ["MLX_OMNI_PROMPT_CACHE_CAPACITY"] = str(args.prompt_cache_capacity)
    if args.prompt_cache_max_tokens is not None:
        os.environ["MLX_OMNI_PROMPT_CACHE_MAX_TOKENS"] = str(args.prompt_cache_max_tokens)
        
    # Start server with uvicorn
    uvicorn.run(
        "mlx_omni_server.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        use_colors=True,
        workers=args.workers,
    )
