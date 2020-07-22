import unittest


class TestLayer(unittest.TestCase):

    def test_layer_1(self):
        from hugectrpy.layers import Dropout, FullyConnected
        d = Dropout(name='dropout1', src_layers=None)
        f = FullyConnected(name='fc1', src_layers=d, n=512)
        print(d)
        print(f)


if __name__ == '__main__':
    unittest.main()
