import unittest


class TestLayer(unittest.TestCase):

    def test_layer_1(self):
        from hugectrpy.layers import Dropout, FullyConnected, ELU, Reshape, Concat
        d = Dropout(name='dropout1', src_layers=None)
        e = ELU(name="elu1", src_layers=d)
        f = FullyConnected(name='fc1', src_layers=e, n=512)
        r = Reshape(name="reshape1", src_layers=f, leading_dim=416)
        c = Concat(name="concat", src_layers=[e, f])
        print(d)
        print(e)
        print(f)
        print(r)
        print(c)


if __name__ == '__main__':
    unittest.main()
