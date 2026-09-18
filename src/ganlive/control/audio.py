"""Finding the machine's audio input. One answer to one question."""
from __future__ import annotations

import sys

#: The lowest-latency host API each platform offers, and the one a multi-channel interface
#: appears on. PortAudio spells them exactly like this.
HOSTAPI = {"win32": "ASIO", "darwin": "Core Audio"}.get(sys.platform, "ALSA")

#: `hostapi=DEFAULT` means `HOSTAPI`; `hostapi=None` means any. Distinguishable, which a
#: plain `None` default is not.
DEFAULT = "<this platform's>"


class NoAudioDevice(RuntimeError):
    """No usable input. Carries the reason, so every caller reports the same one."""


def named_inputs(sd, pattern: str, hostapi: str | None = DEFAULT) -> list[int]:
    """Every input whose name contains `pattern`, on `hostapi` or on any, in device order."""
    hostapi = HOSTAPI if hostapi == DEFAULT else hostapi
    want = pattern.lower()
    return [i for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0 and want in d["name"].lower()
            and (hostapi is None or sd.query_hostapis(d["hostapi"])["name"] == hostapi)]


def starts(sd, device: int, channels: int, samplerate: int = 48000) -> str | None:
    """None if a stream on this device really starts, else why it did not."""
    try:
        stream = sd.InputStream(device=device, channels=channels, samplerate=samplerate,
                                blocksize=256, dtype="float32", callback=lambda *_a: None)
    except Exception as exc:                                      # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    try:
        stream.start()
        stream.stop()
        return None
    except Exception as exc:                                      # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    finally:
        stream.close()


def pick_input(sd, device: int | None = None, pattern: str = "rytm",
               channels: int | None = None, samplerate: int = 48000,
               hostapi: str | None = DEFAULT) -> tuple[int, dict, int]:
    """`(device index, its info, channel count)` for the input to open.

    An explicit `device` is taken as it is, on whatever host API it lives -- a mixer on
    CoreAudio, a loopback on WASAPI -- as long as it starts. Without one, the input is found
    by name on `hostapi`: `DEFAULT` is this platform's multi-channel API (`HOSTAPI`), `None`
    is any of them."""
    hostapi = HOSTAPI if hostapi == DEFAULT else hostapi
    if device is not None:
        info = sd.query_devices(device)
        nch = channels or info["max_input_channels"]
        if nch <= 0:
            raise NoAudioDevice(f"device {device} ({info['name']}) has no inputs")
        why = starts(sd, device, nch, samplerate)
        if why is not None:
            raise NoAudioDevice(f"device {device} ({info['name']}) will not start: {why}")
        return device, info, nch
    if hostapi is not None and hostapi not in [ha["name"] for ha in sd.query_hostapis()]:
        raise NoAudioDevice(
            f"no {hostapi} host API, so no {pattern} can be found by name."
            + (" sounddevice ships ASIO behind SD_ENABLE_ASIO, which must be set before it is"
               " imported." if hostapi == "ASIO" else "")
            + " Pass --audio-device with the input to read, or --audio-name with a name to"
              " look for on any host API.")

    found = named_inputs(sd, pattern, hostapi)
    if not found:
        raise NoAudioDevice(
            f"no input named {pattern}" + (f" on {hostapi}" if hostapi else " on any host API")
            + ". A name in the list is not a connection -- Elektron's Overbridge installer, "
              "for one, registers a node for every product it knows. Run "
              "`ganlive doctor --list` to see what actually opens.")
    tried = []
    for candidate in found:
        info = sd.query_devices(candidate)
        nch = channels or info["max_input_channels"]
        why = starts(sd, candidate, nch, samplerate)
        if why is None:
            return candidate, info, nch
        tried.append(f"  {candidate} {info['name']}: {why}")
    raise NoAudioDevice(
        f"{len(found)} device(s) are named {pattern} and none of them will start. A driver "
        f"may register a node per product it knows, so some of these are machines you do not "
        f"own -- but the real one is failing too, and an exclusive API (ASIO, or WASAPI in "
        f"exclusive mode) gives the device to ONE program at a time: close any DAW or control "
        f"panel holding it. Tried:\n" + "\n".join(tried))
