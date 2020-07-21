import unittest
from hugectrpy.model import Solver


class MyTestCase(unittest.TestCase):

    def test_model_1(self):
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


if __name__ == '__main__':
    unittest.main()
