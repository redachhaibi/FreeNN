from models import *

def test2():
    data_master = DataMaster("MNIST", 512, 8)
    mlp_description = {
        "layer_description":[
            {
                "in_size": data_master.in_size,
                "out_size": 100,
                "bias": True,
                "non_linearity": "ReLU"
            },
            {
                "in_size": 100,
                "out_size": 100,
                "bias": True,
                "non_linearity": "ReLU"
            },
            {
                "in_size": 100,
                "out_size": 100,
                "bias": True,
                "non_linearity": "ReLU"
            },
            {
                "in_size": 100,
                "out_size": data_master.out_size,
                "bias": True,
                "non_linearity": "ReLU"
            },
        ]
    }
    mlp = MLP("from_list", mlp_description)
    mlp.initialize("xavier_uniform_")
    reporter = BatchReporter(2, "weights_gradients")
    trainer = Trainer(mlp, data_master, [reporter])
    trainer.train("CrossEntropyLoss", {"name": "SGD", "lr": 0.001, "momentum":0.9}, 10)

def test1():
    def _i_will_not_write_proper_unit_tests(net):
        inps = torch.rand( (1, 28*28) )
        print( net(inps) )
        print( net.layers )
        print( net.get_weights() )
        print( net.get_biases() )
        print( net.get_weights_gradients() )
        print( net.get_biases_gradients() )

        a = net.get_weights()["layer_0"].std()
        net.initialize("xavier_uniform_")
        b = net.get_weights()["layer_0"].std()

        print( a, b)

    mlp_description = {
        "layer_description":[
            {
                "in_size": 28*28,
                "out_size": 100,
                "bias": True,
                "non_linearity": "ReLU"
            },
            {
                "in_size": 100,
                "out_size": 100,
                "bias": True,
                "non_linearity": "ReLU"
            },
            {
                "in_size": 100,
                "out_size": 100,
                "bias": True,
                "non_linearity": "ReLU"
            },
            {
                "in_size": 100,
                "out_size": 10,
                "bias": True,
                "non_linearity": "ReLU"
            },
        ]
    }
    mlp = MLP("from_list", mlp_description)
    _i_will_not_write_proper_unit_tests(mlp)

def main():
    test1()
    test2()

if __name__ == '__main__':
    main()

