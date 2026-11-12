import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from grl_snam.coph.records import Config,Reading,new_stage
from grl_snam.coph.learner import Learner
from grl_snam.coph.model import tensor,flow,balance
from grl_snam.coph.stage_prediction import FrozenStages,DomainExit


def fixture(mode='smooth'):
    cfg=Config(controller_mode='protected',shaping_mode='progress',replay_domain='retained',replay_objective='mission',stage_blend_mode=mode)
    agent=Learner(cfg,(0.,0.,0.,0.),(4.,0.),mode='evidence')
    agent.stages.append(new_stage(1,(2.1,0.),(1.06,0.,.4,0.),agent.goal,cfg))
    for k,s in enumerate(agent.stages):
        s.weights=tensor(np.linspace(-1.,1.,16)+(2*k))
        s.readings=[Reading('blend:%d'%(5*i+1),i*.12,(float(x),float(y)),.1) for i,(x,y) in enumerate((x,y) for x in np.linspace(.6,1.5,10) for y in (-.5,0.,.5))]
    return agent,FrozenStages.freeze(agent)


def test_index_handoff_is_identical_with_frozen_membership():
    a,atlas=fixture();q=tensor([1.06,.03]);p=a.policy
    assert atlas.advance(q.numpy(),0)==1
    torch.testing.assert_close(atlas.energy(q,0,p),atlas.energy(q,1,p),rtol=0,atol=0)
    x=torch.cat((q,tensor([.3,0.])))
    u0,n0=atlas.control(x,0,p,.85,{},0);u1,n1=atlas.control(x,1,p,.85,{},0)
    torch.testing.assert_close(u0,u1,rtol=0,atol=0);torch.testing.assert_close(n0,n1,rtol=0,atol=0)
    assert abs(balance(x,u0,.85,a.cfg.dt)['residual'])<1e-12


def test_partition_derivatives_are_in_full_gradient():
    a,atlas=fixture();q=tensor([1.03,.03]).requires_grad_();v=atlas.energy(q,0,a.policy);grad=torch.autograd.grad(v,q)[0]
    eps=1e-5
    finite=tensor([(float(atlas.energy(q.detach()+tensor(np.eye(2)[k]*eps),0,a.policy))-float(atlas.energy(q.detach()-tensor(np.eye(2)[k]*eps),0,a.policy)))/(2*eps) for k in range(2)])
    torch.testing.assert_close(grad,finite,atol=1e-7,rtol=1e-6)
    members,weights=atlas.blend_weights(q);frozen=sum(w.detach()*atlas.potential_at(q,0,a.policy,member_ids=[s.id]) for s,w in zip(members,weights))
    wrong=torch.autograd.grad(frozen,q)[0]
    assert torch.linalg.vector_norm(grad-wrong)>1e-5


def test_no_new_support_and_frontier_rejection():
    a,atlas=fixture();q=tensor([2.8,1.8])
    assert not any(s.supported(q.numpy(),a.cfg) for s in atlas.stages)
    with pytest.raises(DomainExit,match='unsupported_blend_state'):atlas.energy(q,1,a.policy)
    before=copy.deepcopy(a.stages[0].readings);atlas.energy(tensor([1.,0.]),0,a.policy)
    assert a.stages[0].readings==before


def test_multistep_handoff_gradient_and_execution_parity():
    a,atlas=fixture()
    def replay(policy,live=False):
        x=tensor([.95,0.,.5,0.]);index=0;loss=tensor(0.)
        for step in range(8):
            index=atlas.advance(x[:2].detach().numpy(),index)
            u,_=atlas.control(x,index,policy,.85,{},step)
            x=flow(x,u,.85,a.cfg.dt);loss=loss+x[:2].square().sum()+.01*u.square().sum()
        return loss,x
    p=a.policy.clone().requires_grad_();loss,_=replay(p);g=torch.autograd.grad(loss,p)[0];eps=1e-5
    fd=tensor([(float(replay(a.policy+tensor(np.eye(2)[k]*eps))[0])-float(replay(a.policy-tensor(np.eye(2)[k]*eps))[0]))/(2*eps) for k in range(2)])
    torch.testing.assert_close(g,fd,atol=1e-6,rtol=1e-5)
    # Live execution uses the same frozen collection and shared controller.
    x=np.array([1.06,.02,.3,0.]);expected,_=atlas.control(tensor(x),1,a.policy,a.damping,{},0)
    actual,_=a.action(x);np.testing.assert_array_equal(actual,expected.detach().numpy())


def test_legacy_default_and_invalid_configuration():
    assert Config().stage_blend_mode=='legacy'
    with pytest.raises(ValueError):Config(stage_blend_mode='unknown')
    with pytest.raises(ValueError):Config(stage_blend_mode='smooth')
    a,atlas=fixture('legacy');q=tensor([1.06,.03]);assert float(abs(atlas.energy(q,1,a.policy)-atlas.energy(q,0,a.policy)))>1e-6


def test_support_membership_jump_is_diagnostic_only():
    a,atlas=fixture();ctx={}
    atlas.control(tensor([1.45,.03,.2,0.]),1,a.policy,.85,ctx,0)
    _,nominal=atlas.control(tensor([1.55,.03,.2,0.]),1,a.policy,.85,ctx,1)
    jump=ctx['last_control']['stage_blend_context_jump']
    assert jump['reason']=='support_membership_changed' and jump['available']
    assert jump['physical_jump']==0.
    assert ctx['last_control']['stage_blend']['field_evaluations']>=1
    fresh={};_,other=atlas.control(tensor([1.55,.03,.2,0.]),1,a.policy,.85,fresh,1)
    torch.testing.assert_close(nominal,other,atol=0,rtol=0)




def test_large_overlap_keeps_center_taper_differentiable():
    a,_=fixture();a.cfg=replace(a.cfg,overlap=.9)
    # Place the second model close enough that the first center has two
    # certified members; otherwise a singleton partition is trivially constant.
    a.stages[1].center=(.3,0.)
    for stage in a.stages:stage.readings=[Reading('large:1',0.,(0.,0.),.1)]
    atlas=FrozenStages.freeze(a);q=tensor([0.,.03]).requires_grad_();v=atlas.energy(q,1,a.policy);g=torch.autograd.grad(v,q)[0]
    eps=1e-6;delta=tensor([eps,0.]);finite=(atlas.energy(q.detach()+delta,1,a.policy)-atlas.energy(q.detach()-delta,1,a.policy))/(2*eps)
    assert abs(float(g[0]-finite))<1e-6
