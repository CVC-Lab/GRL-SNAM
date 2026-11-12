"""Generated grid mazes with hidden doors, screened by the route oracle.

Layouts: 'crossing' (opposite corners, swapped goals) and 'relay' (shared start and
goal, the second agent starts later). Every candidate is predicted by the oracle;
the declared screen keeps those where communication should matter, and the whole
candidate distribution is recorded so selection is visible.
"""
# grl_snam.coph: trimmed to NetworkWorld, NetworkLearner and learner_for, which the hazard demo
# uses. The generated maze family and its oracle screen are not vendored (VENDORED.md).
import copy
from .cooperative_evaluation import CooperativeWorld
from .maze_learning import MazeLearner
from .maze_geometry import corridor_clear


class NetworkWorld(CooperativeWorld):
    def step(self,x,u):
        y,hit,cost,energy=super().step(x,u)
        hit=bool(hit or not corridor_clear(x[:2],y[:2],margin=.12,nodes=self.spec['network_nodes'],edges=self.spec['network_edges']))
        return y,hit,cost,energy


class NetworkLearner(MazeLearner):
    def __init__(self,*args,network,risk_weight=1.,unknown_risk=1.,**kwargs):
        super().__init__(*args,**kwargs);self.navigation['network']=network
        # Non-default task costs only; default runs keep their exact navigation records.
        if risk_weight!=1.:self.navigation['risk_weight']=float(risk_weight)
        if unknown_risk!=1.:self.navigation['unknown_risk']=float(unknown_risk)
        self.stages[0].entry_navigation=copy.deepcopy(self.navigation)


def learner_for(spec):
    # A partial of a module-level class stays picklable for the runner's packet audits.
    from functools import partial
    return partial(NetworkLearner,network=(tuple(map(tuple,spec['network_nodes'])),tuple(map(tuple,spec['network_edges']))))
