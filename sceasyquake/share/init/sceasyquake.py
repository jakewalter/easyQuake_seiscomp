"""
SeisComP kernel init script for sceasyquake.

Installed to $SEISCOMP_ROOT/etc/init/sceasyquake.py by install.sh.
This file makes `seiscomp enable/disable/start/stop/status sceasyquake`
work exactly like any other SeisComP module.
"""

import seiscomp.kernel


class Module(seiscomp.kernel.Module):
    def __init__(self, env):
        seiscomp.kernel.Module.__init__(self, env, env.moduleName(__file__))
