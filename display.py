"""
Howard — Display Control
Handles screen on/off and screensaver disable.
Works on both Pi (Linux) and Windows (stub).
"""

import platform
import subprocess
import os

_IS_WINDOWS = platform.system() == "Windows"


def screen_off():
    if _IS_WINDOWS:
        print("[display] STUB: screen_off (no-op on Windows)")
        return
    _xset("dpms", "force", "off")
    print("[display] Screen off")


def screen_on():
    if _IS_WINDOWS:
        print("[display] STUB: screen_on (no-op on Windows)")
        return
    _xset("dpms", "force", "on")
    _xset("s", "reset")
    print("[display] Screen on")


def disable_screensaver():
    if _IS_WINDOWS:
        print("[display] STUB: disable_screensaver (no-op on Windows)")
        return
    _xset("s", "off")
    _xset("-dpms")
    _xset("s", "noblank")
    print("[display] Screensaver disabled")


def _xset(*args):
    env = {
        **os.environ,
        "DISPLAY": ":0",
        "XAUTHORITY": f"/home/{os.getenv('USER', 'pi')}/.Xauthority"
    }
    subprocess.run(["xset", *args], env=env, check=False)