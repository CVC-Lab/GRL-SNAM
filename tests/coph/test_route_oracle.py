import unittest
from grl_snam.coph.crossing_demo import specification,predicted,ARMS
from grl_snam.coph.route_oracle import predict,true_distance


class RouteOracleTests(unittest.TestCase):
    def setUp(self):self.spec=specification(76000)

    def test_reproduces_hand_derived_crossing_predictions(self):
        for arm in ARMS:
            o=predict(self.spec,arm)['agents'];p=predicted(arm)
            for i in (0,1):self.assertAlmostEqual(o[i]['wasted'],p[i]['wasted'],delta=.15,msg=(arm,i))

    def test_delay_removes_superadditivity_then_all_benefit(self):
        none=predict(self.spec,'disabled')['team_wasted']
        saved=lambda arm,d:none-predict(self.spec,arm,delay=d,speed=.3)['team_wasted']
        self.assertGreater(saved('reciprocal_fixed',0),saved('one_way_10',0)+10)
        self.assertAlmostEqual(saved('reciprocal_fixed',12),saved('one_way_10',12),delta=.2)
        self.assertAlmostEqual(saved('reciprocal_fixed',40),0,delta=.2)

    def test_true_distance_matches_routes(self):
        self.assertAlmostEqual(true_distance(self.spec,[0.,4.],[12.,4.]),20.)
        self.assertAlmostEqual(true_distance(self.spec,[5.,4.2],[12.,4.]),25.)


if __name__=='__main__':unittest.main()
