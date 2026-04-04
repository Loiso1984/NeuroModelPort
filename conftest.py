import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

collect_ignore_glob = [
    "tests/archive/*",
    "tests/utils/*",
    "tests/branches/*",
]
