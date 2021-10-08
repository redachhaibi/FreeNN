import torch
import torchvision
import torchvision.transforms as transforms

import numpy

from fiberedae.utils import nn as nnutils
from icecream import ic

def _get_attr(obj, attr_name, human_err_message):
    try:
        thing = getattr(obj, attr_name)
    except (AttributeError, torch.nn.modules.module.ModuleAttributeError):
        raise ValueError(human_err_message.format(attr_name=attr_name))
    return thing

class DataMaster:
    def __init__(self, dataset_name, mininatch_size, num_workers=8):
    
        self.mininatch_size = mininatch_size
        self.num_workers = num_workers

        self.trainset = None
        self.trainloader = None
        self.testset = None
        self.testloader = None
        self.classes = None
        self.in_size = None
        self.out_size = None
        
        fct = _get_attr(self, "load_"+ dataset_name.lower(), "There's no dataset by the name: {attr_name}")
        fct()

    def load_cifar10(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.mininatch_size, shuffle=True, num_workers=self.num_workers)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.mininatch_size, shuffle=False, num_workers=self.num_workers)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.in_size = 32*32*3
        self.out_size =  10

    def load_mnist(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.mininatch_size, shuffle=True, num_workers=self.num_workers)
        self.testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.mininatch_size, shuffle=False, num_workers=self.num_workers)
        self.classes = range(10)
        self.in_size = 28*28
        self.out_size =  10

class MLP(torch.nn.Module):
    def __init__(self, build_name, build_kwargs):
        """build_name is fonction from self that starts with _build_, build_kwargs is a dict for the arguments given to that function"""
        super(MLP, self).__init__()
    
        fct = _get_attr(self, "_build_" +  build_name.lower(), "There's no build named: {attr_name}")
        fct(**build_kwargs)

        for parameter in self.parameters():
            parameter.requires_grad_(True)
            parameter.retain_grad()

    def _reset(self):
        self.layers = []

    def _get_non_linearity(self, non_linearity_name):
        return _get_attr(torch.nn, non_linearity_name, "There's no non_linearity named: {attr_name}")

    def _append_layer(self, layer, non_linearity_name):
        self.layers.append(layer)
        non_lin = self._get_non_linearity(non_linearity_name)()
        self.layers.append(non_lin)

    def _build_from_list(self, layer_description:list):
        """expects a list of dicts,each with keys: in_size, out_size, bias, non_linearity"""
        self._reset()
        for elmt in layer_description:
            layer = torch.nn.Linear(elmt["in_size"], elmt["out_size"], bias=elmt["bias"])
            self._append_layer(layer, elmt["non_linearity"])

        self.layers = torch.nn.Sequential(*self.layers)

    def _build_procedural(self, in_size, hid_size, out_size, nb_hidden, non_linearity="ReLU", last_non_linearity="LogSoftmax", bias=True):
        """procedurally builds an MLP with all hiddens the same size"""
        self._reset()
        if nb_hidden == 0 :
            layer = torch.nn.Linear(in_size, out_size, bias=bias)
            self._append_layer(layer, non_linearity)        
        else:
            layer_first = torch.nn.Linear(in_size, hid_size, bias=bias)
            self._append_layer(layer_first, non_linearity)        
            
            for _ in range(nb_hidden):
                layer = torch.nn.Linear(hid_size, hid_size, bias=bias)
                self._append_layer(layer, non_linearity)        
            
            layer_last = torch.nn.Linear(hid_size, out_size, bias=bias)
            self._append_layer(layer_last, last_non_linearity)        

        self.layers = torch.nn.Sequential(*self.layers)

    def initialize(self, name, torch_kwargs={}):
        """name must be in torch.nn.init"""
        init = _get_attr(torch.nn.init, name, "torch.nn.init has no initialization named: {attr_name}")
        for layer in self.layers:
            try:
                init(layer.weight, **torch_kwargs)
            except torch.nn.modules.module.ModuleAttributeError:
                pass

    def forward(self, x_inputs):
        x_inputs = torch.flatten(x_inputs, 1)
        return self.layers(x_inputs)

    def _get_data(self, attr_name):
        ret = {}
        for aidi, layer in enumerate(self.layers):
            attrs = attr_name.split(".")
            obj = layer
            add_it = True
            for attr in attrs:
                try:
                    obj = getattr(obj, attr)
                except torch.nn.modules.module.ModuleAttributeError:
                    add_it = False
                    break
    
            if add_it:
                ret["layer_%s" % aidi] = obj

        return ret

    def get_weights(self):
        """returns a dict of weights, one entry per layer"""
        return self._get_data("weight")

    def get_biases(self):
        """returns a dict of biases, one entry per layer"""
        return self._get_data("bias")
    
    def get_weights_gradients(self):
        """returns a dict of weights, one entry per layer"""
        return self._get_data("weight.grad")
    
    def get_biases_gradients(self):
        """returns a dict of biases, one entry per layer"""
        return self._get_data("bias.grad")

class BatchReporter:
    def __init__(self, period, variable_name):
        self.period = period
        self.variable_name = variable_name
        self.current_tick = 0
        self.dir = None
        self.mkdir()

    def mkdir(self):
        import time
        import os

        fix = time.ctime().replace(":", "-").replace(" ", "_")
        self.dir = self.variable_name + "_" + fix
        print("making folder:", self.dir)
        os.mkdir(self.dir)

    def tick(self, model):
        if self.current_tick % self.period == 0:
            self.save_report(model)
        self.current_tick += 1

    def save_report(self, model):
        import os

        fct_name = "get_" +  self.variable_name.lower()
        fct = _get_attr(model, fct_name, "Model has no function: {attr_name}")

        for key, value in fct().items():
            path = os.path.join( self.dir, key)
            if not os.path.exists(path):
                os.mkdir(path)

            filename = os.path.join( path, "%s.npy" % self.current_tick ) 
            value = value.cpu().detach().numpy()
            numpy.save(filename, value)

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, net, loss_name, optimizer_kwargs):
        """loss_name should be in torch.nn, optimizer kwargs is a dict with a field name that must in torcj.optim. The rest must be the parameters for the optimizer"""
        super(Trainer, self).__init__()

        self.net = net
        self.criterion = _get_attr(torch.nn, loss_name, "There's no loss: {attr_name}")()
        
        optimizer_fct = _get_attr(torch.optim, optimizer_kwargs["name"], "There's no optimizer: {attr_name}")
        del optimizer_kwargs["name"]
        self.optimizer = optimizer_fct(self.net.parameters(), **optimizer_kwargs)

    def _one_pass(self, data, reporter, training):
        self.optimizer.zero_grad()

        inputs, labels = data
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        if training:
            loss.backward(retain_graph = True)
            self.optimizer.step()
            reporter.tick(self.net)

        return loss.item()

    def train(self, nb_epochs, data_master, reporter):
        from tqdm import trange

        learnin_curves = {
            "train": [],
            "test": []
        }

        pbar = trange(nb_epochs)
        for epoch in pbar:
            train_loss = 0.0
            for train_batch_id, data in enumerate(data_master.trainloader):
                train_loss += self._one_pass(data, reporter=reporter, training=True)
            train_loss = train_loss / (train_batch_id+1)
            learnin_curves["train"].append(train_loss)

            test_loss = 0.0
            for test_batch_id, data in enumerate(data_master.testloader):
                test_loss += self._one_pass(data, reporter=None, training=False)
            test_loss = test_loss / (test_batch_id+1)
            learnin_curves["test"].append(test_loss)
                            
            label = "epoch: %d, train: %.4f, test: %.4f" % (epoch, train_loss, test_loss)
            pbar.set_description( label )

        return learnin_curves

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
    trainer = Trainer(mlp, "CrossEntropyLoss", {"name": "SGD", "lr": 0.001, "momentum":0.9})
    trainer.train(10, data_master, reporter)

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
    # test1()
    test2()

if __name__ == '__main__':
    main()


