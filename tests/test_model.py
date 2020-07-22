import unittest


class TestModel(unittest.TestCase):

    def test_model_1(self):
        from hugectrpy.model import Solver
        s = Solver()
        #print(s)

    def test_model_2(self):
        from hugectrpy.model import AdamOptimizer
        a = AdamOptimizer()
        #print(a)

    def test_model_3(self):
        from hugectrpy.model import MomentumSGD
        m = MomentumSGD()
        #print(m)

    def test_model_4(self):
        from hugectrpy.model import Nesterov
        n = Nesterov()
        #print(n)

    def test_model_5(self):
        from hugectrpy.model import Solver, MomentumSGD, Model
        solver = Solver()
        m = MomentumSGD()

        model = Model(solver, m, None)
        #print(model)

    def test_model_criteo(self):
        from hugectrpy.model import Solver, AdamOptimizer, Model
        solver = Solver()
        optimizer = AdamOptimizer(alpha=0.005)
        model = Model(solver, optimizer)

        from hugectrpy.layers import Dense, Label, Sparse, Data, DistributedSlotSparseEmbeddingHash, \
            Reshape, FullyConnected, RELU, BinaryCrossEntropyLoss

        dense = Dense(name='dense', dim=0)
        label = Label(name='label', dim=1)
        sparse = Sparse(name='data1', max_feature_num_per_sample=100, slot_num=1)
        data = Data(name='data', source="./file_list.txt",
                    eval_source="./file_list_test.txt",
                    check="Sum", label=label, dense=dense, sparse=sparse)
        model.add_layer(data)
        emb = DistributedSlotSparseEmbeddingHash(name='sparse_embedding1',
                                                 src_layers=sparse, vocabulary_size=1603616,
                                                 load_factor=0.75, embedding_vec_size=64, combiner=0)

        re1 = Reshape(name='reshape1', src_layers=emb, leading_dim=64)

        fc1 = FullyConnected(name='fc1', src_layers=re1, n=200)
        relu1 = RELU(name='relu1', src_layers=fc1)
        fc2 = FullyConnected(name="fc2", src_layers=relu1, n=200)
        relu2 = RELU(name='relu2', src_layers=fc2)
        fc3 = FullyConnected(name='fc3', src_layers=relu2, n=200)
        relu3 = RELU(name='relu3', src_layers=fc3)
        fc4 = FullyConnected(name='fc4', src_layers=relu3, n=1)
        loss = BinaryCrossEntropyLoss(name='loss', src_layers=[fc4, label])

        model.add_layer_re(loss)

        print(model)


if __name__ == '__main__':
    unittest.main()
