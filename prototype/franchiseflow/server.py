"""Single-process WSGI entry point for local/container deployments."""

from app import app, _recover_jobs
from runtime_health import server_bind


def main() -> None:
    from waitress import serve

    host, port = server_bind()
    _recover_jobs()
    # Keep exactly one application process while local JSON and worker locks
    # are used. Waitress threads handle polling/downloads during a render.
    serve(app, host=host, port=port, threads=4)


if __name__ == "__main__":
    main()
