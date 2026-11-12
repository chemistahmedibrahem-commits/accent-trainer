import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="accent-test-")
os.environ.setdefault("AT_ACOUSTIC_BACKEND", "synthetic")
os.environ.setdefault("AT_DATABASE_URL", f"sqlite+aiosqlite:///{_tmp}/test.db")
os.environ.setdefault("AT_AUDIO_DIR", f"{_tmp}/audio")
