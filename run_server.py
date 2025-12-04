# run_server.py
# Convenience runner so you can start the FastAPI app with:
#   python3 run_server.py
# It wraps uvicorn.run and reads env vars HOST, PORT, RELOAD for quick overrides.

import os
import sys

if __name__ == "__main__":
    try:
        import uvicorn
    except Exception as e:
        print("Missing dependency 'uvicorn'. Install with: python3 -m pip install uvicorn")
        raise

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "3000"))
    reload_flag = os.getenv("RELOAD", "true").lower() in ("1", "true", "yes")

    # Use module path to the FastAPI app object
    module = "src.app:app"

    # Print helpful info
    print(f"Starting Uvicorn server for {module} on {host}:{port} (reload={reload_flag})")

    uvicorn.run(module, host=host, port=port, reload=reload_flag)
