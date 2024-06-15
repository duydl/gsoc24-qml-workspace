"""
Quantum autoencoder implementations using pennylane
"""
import pennylane as qml
import torch
import qutip

class ContrastiveConvEncoderCircuit:
    def __init__(self, input_qbits, latent_qbits, aux_qbits, device, img_dim, kernel_size, stride, DRCs, diff_method="best"):
        """Create basic SQAE

        Args:
            input_qbits (int): number of qbits to upload input and use as encoder
            latent_qbits (int): number of latent qbits
            aux_qbits (int): number of latent qbits
            device (pennylane device): pennylane device to use for circuit evaluation
            img_dim (int): dimension of the images (width)
            kernel_size (int): size of the kernel to use when uploading the input
            stride (int): stride to use when uploading the input
            DRCs (int): number of times to repeat the encoding upload circuit in the encoder
            diff_method (str): method to differentiate quantum circuit, usually "adjoint" is best.
        """

        self.dev = device
        self.input_qbits = input_qbits
        self.latent_qbits = latent_qbits
        self.aux_qbits = aux_qbits # Currently only one aux qbit is used in any case
        self.total_qbits = input_qbits * 2 + aux_qbits
        self.trash_qbits = input_qbits - latent_qbits
        
        self.circuit_node = qml.QNode(self.circuit, device, interface="torch", diff_method=diff_method)

        self.kernel_size = kernel_size
        self.stride = stride
        self.DRCs = DRCs

        self.input_shape = (img_dim, img_dim)
        self.number_of_kernel_uploads = len(list(range(0, img_dim - kernel_size + 1, stride)))**2
        self.parameters_shape = (DRCs * 2 * self.number_of_kernel_uploads * kernel_size ** 2,)
        self.params_per_layer = self.parameters_shape[0] // DRCs

    def _conv_upload(self, params, img, kernel_size, stride, wire=0):
        """Upload image using the convolution like method

        Args:
            params (list): parameters to use
            img (2d list): the actual image
            kernel size (int): kernel size for upload
            stride (int): stride for upload
            wires (list): list of integers to use as qbit index for upload
        """
        def single_upload(params, data, wire):
            """Upload data on a single qbit

            Args:
                params (list): parameters to use for upload, must be twice as long as data
                data (list): data to upload
                wire (int): on which wire to upload
            """
            data_flat = data.flatten()
            for i, d in enumerate(data_flat):
                if i % 3 == 0:
                    qml.RZ(params[i * 2] + params[i * 2 + 1] * d, wires=wire)
                elif i % 3 == 1:
                    qml.RY(params[i * 2] + params[i * 2 + 1] * d, wires=wire)
                elif i % 3 == 2:
                    qml.RZ(params[i * 2] + params[i * 2 + 1] * d, wires=wire)
        
        number_of_kernel_uploads = len(list(range(0, img.shape[1]-kernel_size+1, stride))) * len(list(range(0, img.shape[0]-kernel_size+1, stride)))

        params_per_upload = len(params) // number_of_kernel_uploads

        upload_counter = 0

        for y in range(0, img.shape[1]-kernel_size+1, stride):
            for x in range(0, img.shape[0]-kernel_size+1, stride):
                single_upload(params[upload_counter * params_per_upload: (upload_counter + 1) * params_per_upload],
                                   img[x:x+kernel_size, y:y+kernel_size], wire)
                upload_counter = upload_counter + 1
                wire = wire + 1

    def encoder(self, params, inputs, wires):
        """The encoder circuit for the SQAE

        Args:
            params (list): parameters to use
            inputs (list): inputs to upload
        """
        def circular_entanglement(wires):
            qml.CNOT(wires=[wires[-1], wires[0]])
            for i in wires[:-1]:
                qml.CNOT(wires=[i, i+1])

        for i in range(self.DRCs):
            self._conv_upload(
                params[i * self.params_per_layer:(i + 1) * self.params_per_layer], 
                inputs, self.kernel_size, self.stride, 
                wire=wires[0]
                )
            circular_entanglement(wires)

    def circuit(self, params, input_1, input_2):
        """Full circuit to be used as Constrastive CNN

        Includes two encoders and SWAP test

        Args:
            params (list): shared parameters for two encoding PQC
            input_1 (list): image 1 input
            input_2 (list): image 2 input

        Returns:
            expectation value of readout bit

        """
        wire_1 = list(range(self.number_of_kernel_uploads))
        wire_2 = list(range(self.number_of_kernel_uploads, self.number_of_kernel_uploads*2))
        self.encoder(params, input_1, wire_1)
        self.encoder(params, input_2, wire_2)

        # SWAP test
        qml.Hadamard(wires=self.total_qbits - 1)
        for i in range(self.latent_qbits):
            qml.CSWAP(wires=[self.total_qbits - 1, 0 + i, self.number_of_kernel_uploads + i])
        qml.Hadamard(wires=self.total_qbits - 1)

        return qml.expval(qml.PauliZ(self.total_qbits - 1))

    def plot_circuit(self):
        """Plots the circuit with dummy inputs
        """

        input2 = torch.rand(self.input_shape)
        input1 = torch.rand(self.input_shape)
        params = torch.rand(self.parameters_shape)
        fig, _ = qml.draw_mpl(self.circuit_node)(params, input1, input2)
        fig.show()

    def draw_circuit(self):
        """Draw text-based circuit to check if similar values are uploaded to the two encoders
        """

        inputs = torch.rand(self.input_shape)
        params = torch.rand(self.parameters_shape)
        print(self.circuit_node(params, inputs, inputs))
        print(qml.draw(self.circuit)(params, inputs, inputs))

