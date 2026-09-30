"""Allow `python -m n_atlas_client`."""

import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main())