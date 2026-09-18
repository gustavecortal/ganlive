"""The commands `ganlive` dispatches to. One module each, each with `main(argv)`.

**Two pieces of process setup live here, and they have to run before any subcommand's own
imports.** Importing this package is what runs them, and Python runs a package's `__init__`
before any module inside it -- so `ganlive.cli`'s `import_module`, a direct
`python -m ganlive.tools.play`, and a test importing one of these all get them, in that
order, with nothing to remember.

They were copied into the tools themselves, six of the ten, each above a block of imports
that then needed `# noqa: E402` to say why. Four copies of one line and three of the other,
and the three had already drifted: one reconfigured stdout and left stderr to raise, and two
of the three lacked the `hasattr` guard.
"""

import os
import sys

#: `sounddevice` ships ASIO behind this and reads it **at import**, so it is no use to a tool
#: that sets it after the import -- and ASIO is the only host API a Rytm's stems appear on.
#: Ignored everywhere else, which is why it is unconditional rather than `sys.platform`-gated.
os.environ.setdefault("SD_ENABLE_ASIO", "1")

# A Windows console is cp1252 by default, and the dynamo exporter prints a check mark when it
# succeeds -- so an export used to die with a `UnicodeEncodeError` *after* doing all of the
# work. Both streams, because a traceback goes to the other one.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
