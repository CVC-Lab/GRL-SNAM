"""What a sender transmits about its learned environment, on one shared channel.

Every mode keeps the same packet schedule, validation and byte accounting; only the
environmental content of the anchors changes:

- field: the sender's learned graded environmental field (V_env values, gradients, bootstrap covariance);
- occupancy(tau): the same anchor points, values quantized to blocked/free, no gradient;
- decision: occupancy at the sender's own avoidance threshold 1/(4 w_sender);
- raw: the sender's raw sensor readings, no learned model, no gradient information.

`NoLearningReceiver` receives field packets but never fits them into its own environmental field;
they still reach its route store.
"""
from dataclasses import replace
import numpy as np
from .maze_family import NetworkLearner

BLOCKED=20.  # the route scorer's clipping ceiling: an impassable-looking cell
GRADIENT_UNKNOWN=1e4  # variance for gradients a mode does not observe


def quantized(packet,tau):
    anchors=tuple((*a[:2],BLOCKED if a[2]>=tau else 0.,0.,0.) for a in packet.anchors);n=len(anchors)
    return replace(packet,anchors=anchors,covariance=tuple(map(tuple,np.diag([.0025,GRADIENT_UNKNOWN,GRADIENT_UNKNOWN]*n))),uncertainty=f'occupancy tau={tau}')


class FieldLearner(NetworkLearner):
    mode='field'


class OccupancyLearner(NetworkLearner):
    mode='occupancy'
    def __init__(self,*args,tau,**kwargs):
        super().__init__(*args,**kwargs);self.tau=float(tau)
    def export(self,x,recipient):
        m=super().export(x,recipient);return None if m is None else quantized(m,self.tau)


class DecisionLearner(NetworkLearner):
    """Shares the sender's verdict: avoid where its own risk cost per metre exceeds its time cost."""
    mode='decision'
    def export(self,x,recipient):
        m=super().export(x,recipient)
        return None if m is None else replace(quantized(m,1/(4*self.navigation.get('risk_weight',1.))),uncertainty='decision')


class RawLearner(NetworkLearner):
    """Raw readings at the most hazardous supported points, value only."""
    mode='raw'
    def export(self,x,recipient):
        m=super().export(x,recipient)
        if m is None:return None
        s=self.stages[-1];picked=[]
        for r in sorted(s.readings,key=lambda r:(-r.value,r.uid)):
            if int(r.uid.split(':')[-1])%5 and all(np.linalg.norm(np.asarray(r.q)-p)>.3 for p in picked):picked.append(np.asarray(r.q))
            if len(picked)==self.communication.anchors:break
        values={tuple(r.q):r.value for r in s.readings}
        anchors=tuple((float(q[0]),float(q[1]),float(values[tuple(q)]),0.,0.) for q in picked)
        if not anchors:return m
        n=len(anchors)
        return replace(m,anchors=anchors,covariance=tuple(map(tuple,np.diag([.0025,GRADIENT_UNKNOWN,GRADIENT_UNKNOWN]*n))),uncertainty='raw readings')


class NoLearningReceiver(NetworkLearner):
    mode='no_learning'
    def _field_fit(self):
        # Received fields never enter this agent's own field fit; the route store still sees them.
        saved=self.received;self.received={}
        try:return super()._field_fit()
        finally:self.received=saved


class LedgerExport:
    """Each packet summarizes one retained stage in turn (newest first), not only the newest.

    One implementation of ledger export; the formulation's Algorithm B encodes the committed
    ledger but prescribes no rotation schedule. Bounds, support and stage identity come from the
    summarized stage, so receiver validation is unchanged.
    """
    def export(self,x,recipient):
        turn=getattr(self,'_ledger_turn',0);self._ledger_turn=turn+1
        order=self.stages;chosen=order[-1-turn%len(order)]
        self.stages=[s for s in order if s is not chosen]+[chosen]
        self._summarized=chosen if chosen is not order[-1] else None
        try:return super().export(x,recipient)
        finally:self.stages=order;self._summarized=None

    def anchor_candidates(self,x,stage):
        if getattr(self,'_summarized',None) is not stage:return super().anchor_candidates(x,stage)
        # An earlier stage is summarized at its own supported measurement locations.
        return [np.asarray(r.q) for r in stage.readings if int(r.uid.split(':')[-1])%5]


MODES=dict(field=FieldLearner,occupancy=OccupancyLearner,decision=DecisionLearner,raw=RawLearner,no_learning=NoLearningReceiver)
LEDGER_MODES={k:type(f'Ledger{v.__name__}',(LedgerExport,v),{}) for k,v in MODES.items()}
# Module-level names keep the generated classes picklable for the runner's packet audits.
globals().update({cls.__name__:cls for cls in LEDGER_MODES.values()})
