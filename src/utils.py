import os
import tempfile
from contextlib import contextmanager
from dotenv import load_dotenv


load_dotenv()


FILLERS = [
"um", "uh", "er", "ah", "like", "you know", "kind of", "sort of", "so",
"actually", "literally", "basically"
]


IDEAL_WPM_RANGE = (120, 160)


@contextmanager
def named_tempfile(suffix: str = ""):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        f.close()
        yield f.name
    finally:
        try:
            os.unlink(f.name)
        except FileNotFoundError:
            pass




def safe_get_env(key: str, default: str = ""):
    return os.getenv(key, default)