# examples/volrover_lab_demo.py — run the GRL-SNAM lab in a LIVE volrover3.
#
# A thin wrapper over grl_snam_lab.run_in_volrover() (the real entry point). Run
# it INSIDE volrover3's embedded Python console:
#
#   * REPL tab:        exec(open(".../examples/volrover_lab_demo.py").read())
#   * or simply, in the REPL:   import grl_snam_lab; grl_snam_lab.run_in_volrover()
#   * or Jobs tab -> "Load Script..." and pick this file.
#
# LAYERING: volrover3 does NOT depend on GRL-SNAM. This lives here and is run by
# volrover3's GENERIC script runner only when GRL-SNAM is installed in the
# interpreter's environment. `vrhost` is injected by the volrover3 host at runtime
# (a soft dependency); outside volrover3 run_in_volrover() explains that.

from grl_snam_lab import run_in_volrover

run_in_volrover()
