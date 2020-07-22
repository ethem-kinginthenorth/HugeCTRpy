import unittest


class TestModel(unittest.TestCase):

    def test_model_1(self):
        from hugectrpy.model import Solver
        s = Solver()
        print(s)

    def test_model_2(self):
        from hugectrpy.model import AdamOptimizer
        a = AdamOptimizer()
        print(a)

    def test_model_3(self):
        from hugectrpy.model import MomentumSGD
        m = MomentumSGD()
        print(m)

    def test_model_4(self):
        from hugectrpy.model import Nesterov
        n = Nesterov()
        print(n)

    def test_model_5(self):
        from hugectrpy.model import Solver, MomentumSGD, Model
        solver = Solver()
        m = MomentumSGD()

        model = Model(solver, m, None)
        print(model)


if __name__ == '__main__':
    unittest.main()
