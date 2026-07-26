# examples/volrover_lab_animated.py — the GRL-SNAM agent WALKS its track in a
# LIVE volrover3 window, animated.
#
# Load it from volrover3's Jobs tab -> "Load Script..." -> "Run as Job". The
# scheduler calls step(dt) each tick; the agent advances along the closed track
# by MUTATING its node transform — the terrain, the track, and the agent's
# geometry are built ONCE and never rebuilt, so this is a true in-place animation
# (the whole point of the direct-wrapped Scene API: move coordinates, don't
# destroy+recreate). Pause / stop / adjust speed from the Jobs tab.
#
# LAYERING: volrover3 does NOT depend on GRL-SNAM. This runs under volrover3's
# GENERIC job runner only when GRL-SNAM is installed in the interpreter's
# environment. `vrhost` is injected by the volrover3 host at runtime.

from grl_snam_lab.demo import agent_position, demo_scene
from grl_snam_lab.lab import Lab

try:
    import vrhost  # host-injected inside volrover3; a soft dependency
except ImportError as exc:  # pragma: no cover - only meaningful inside volrover3
    raise RuntimeError(
        "volrover_lab_animated: `vrhost` not found — load this INSIDE volrover3's "
        "embedded Python (Jobs tab -> Load Script -> Run as Job). For a standalone "
        "still image use `grl-snam-lab-demo out.png`."
    ) from exc

# One-time setup (runs once when the job is submitted): adopt the host's live app
# + SceneGraph and build the demo. demo_scene adds the agent marker ONCE.
_lab = demo_scene(Lab(app=vrhost.app(), scene=vrhost.scene()))
_t = 0.0
_SPEED = 1.0  # time multiplier; raise to walk faster

print(
    "grl_snam_lab: animated agent job loaded — the red marker will walk the "
    "yellow track over the terrain. Pause/stop it from the Jobs tab."
)


def step(dt):
    """Scheduler tick (dt = seconds since the last tick): advance time and move
    the agent to its new position on the track. Only the ``agent0`` node's
    transform changes — nothing is added or removed, so num_nodes stays constant.
    """
    global _t
    _t += dt * _SPEED
    # animate_agent would also pump; do it directly so the intent is explicit:
    _lab.move("agent0", *agent_position(_t))
    _lab.pump()
