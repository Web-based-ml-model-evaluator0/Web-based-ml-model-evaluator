"""
Entrypoint wrapper for Web Based ML Model Evaluator.
Redirects execution to src.app.main() to preserve root runnable convenience.
"""

from src.app import main

if __name__ == "__main__":
    main()
