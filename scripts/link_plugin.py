"""Link the plugin folder into a QGIS profile, for development.

    python scripts/link_plugin.py            # the default profile
    python scripts/link_plugin.py --profile other --remove

QGIS then loads vineliner/ straight from this checkout: enable it once under
"Plugins > Manage and Install Plugins > Installed", and after editing the code
reload it with the Plugin Reloader plugin. On Windows the link is a directory
junction (no administrator rights needed); elsewhere a symbolic link.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def plugins_dir(profile: str) -> Path:
    if sys.platform == "win32":
        base = Path(os.environ["APPDATA"]) / "QGIS" / "QGIS3"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "QGIS" / "QGIS3"
    else:
        base = Path.home() / ".local" / "share" / "QGIS" / "QGIS3"
    return base / "profiles" / profile / "python" / "plugins"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--remove", action="store_true", help="remove the link instead")
    args = parser.parse_args(argv)
    target = plugins_dir(args.profile) / "vineliner"
    if args.remove:
        if target.is_symlink() or target.is_junction():
            target.unlink() if target.is_symlink() else os.rmdir(target)
            print(f"removed {target}")
        else:
            print(f"no link at {target}")
        return 0
    if target.exists():
        print(f"{target} already exists; remove it first (--remove removes only a link)")
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(target), str(ROOT / "vineliner")], check=True)
    else:
        target.symlink_to(ROOT / "vineliner", target_is_directory=True)
    print(f"linked {target} -> {ROOT / 'vineliner'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
