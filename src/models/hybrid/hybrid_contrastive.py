import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from pytorch_metric_learning import losses

import sys
sys.path.append("../../")
from utils.utils import get_preprocessing

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}
_POOLING = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}

class ConvUnit(nn.Module):
    """
    A convolutional unit consisting of a convolutional layer, batch normalization, activation, and pooling.
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, pool_type, pool_kernel_size, pool_stride, activ_type):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.activ = _ACTIVATIONS[activ_type]()
        self.pool = _POOLING[pool_type](kernel_size=pool_kernel_size, stride=pool_stride)

    def forward(self, x):
        return self.pool(self.activ(self.bn(self.conv(x))))

class MNISTConvEncoder(nn.Module):
    """
    A convolutional encoder for MNIST images.
    """
    backbone_output_size = 196
    output_size = 8
    def __init__(self, activ_type, pool_type):
        super().__init__()
        self.conv_unit1 = ConvUnit(1, 4, 3, 1, 1, pool_type, 2, 2, activ_type)
        self.conv_unit2 = ConvUnit(4, 4, 3, 1, 1, pool_type, 2, 2, activ_type)
        self.fc1 = nn.Linear(self.backbone_output_size, self.output_size)

    def forward(self, x):
        x = self.conv_unit1(x)
        x = self.conv_unit2(x)
        x = x.view(-1, self.backbone_output_size)
        return self.fc1(x)

import pennylane as qml
class QuantumHead(nn.Module):
    def __init__(self, in_features, out_features, n_qlayers):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_qubits = [in_features,out_features][in_features<out_features]
        self.n_qlayers = n_qlayers
        
        self.device = qml.device('default.qubit', wires=self.n_qubits)
        @qml.qnode(self.device, interface='torch')
        def quantum_circuit(inputs, weights):
            # print(inputs.shape)
            qml.templates.AngleEmbedding(inputs, wires=range(self.in_features))
            
            # Apply layers of rotation gates and CNOTs for entanglement
            for layer in range(n_qlayers):
                for qubit in range(self.n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                
            return [qml.expval(qml.PauliZ(i)) for i in range(self.out_features)]
        
        self.quantum_circuit = quantum_circuit
        dummy_inputs = torch.randn(self.in_features)
        dummy_weights = torch.randn(self.n_qlayers, self.n_qubits, 3)
        print(qml.draw(self.quantum_circuit)(dummy_inputs, dummy_weights))
        
        self.quantum_layer = qml.qnn.TorchLayer(self.quantum_circuit, {"weights": (n_qlayers, self.n_qubits, 3)})
        
        # batch_dim = 8
        # x = torch.zeros((batch_dim, self.in_features))
        # print(self.quantum_layer(x).shape)
        
    def forward(self, inputs):
        q_outputs = self.quantum_layer(inputs)
        return q_outputs

class MNISTQSupContrast(pl.LightningModule):
    """
    A PyTorch Lightning module for supervised contrastive learning on the MNIST dataset.
    """
    def __init__(self, activ_type, pool_type, head_output, n_qlayers, lr, pos_margin=0.25, neg_margin=1.5, preprocess=None):
        super().__init__()
        self.save_hyperparameters()
        self.preprocessing = get_preprocessing(preprocess)
        self.encoder = MNISTConvEncoder(activ_type, pool_type)
        self.head = QuantumHead(MNISTConvEncoder.output_size, head_output, n_qlayers)
        self.loss = losses.ContrastiveLoss(pos_margin=pos_margin, neg_margin=neg_margin)
        self.train_loss = torchmetrics.MeanMetric()
        self.valid_loss = torchmetrics.MeanMetric()

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.preprocessing:
            x = self.preprocessing(x)
        embeddings = self.forward(x)
        loss = self.loss(embeddings, y)
        self.train_loss.update(loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.preprocessing:
            x = self.preprocessing(x)
        embeddings = self.forward(x)
        loss = self.loss(embeddings, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
