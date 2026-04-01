import os
import re
from typing import Iterable, Optional


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1].strip()
    return value


def _read_key_from_bashrc(key_names: Iterable[str], bashrc_path: Optional[str] = None) -> Optional[str]:
    """
    Read API keys from ~/.bashrc style lines:
      export KEY=value
      KEY=value
    """
    path = bashrc_path or os.path.expanduser("~/.bashrc")
    if not os.path.isfile(path):
        return None

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        for name in key_names:
            pattern = rf"^(?:export\s+)?{re.escape(name)}\s*=\s*(.+)$"
            m = re.match(pattern, line)
            if not m:
                continue
            value = _strip_quotes(m.group(1))
            if value:
                return value
    return None


def resolve_api_key(
    explicit_key: Optional[str] = None,
    key_names: Iterable[str] = ("DEEPSEEK_API_KEY", "FALCON_API_KEY"),
    bashrc_path: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve API key in this order:
      1) explicit argument (CLI)
      2) environment variables
      3) ~/.bashrc
    """
    if explicit_key and str(explicit_key).strip():
        return str(explicit_key).strip()

    for name in key_names:
        value = os.environ.get(name)
        if value and str(value).strip():
            return str(value).strip()

    return _read_key_from_bashrc(key_names, bashrc_path=bashrc_path)
