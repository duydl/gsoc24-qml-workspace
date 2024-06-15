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
        
        self.circuit_node = qml.QNode(self.circuit, device, interface="torch")

        self.kernel_size = kernel_size
        self.stride = stride
        self.DRCs = DRCs

        self.input_shape = (img_dim, img_dim)
        self.number_of_kernel_uploads = len(list(range(0, img_dim - kernel_size + 1, stride)))**2
        self.parameters_shape = (DRCs * 2 * self.number_of_kernel_uploads * kernel_size ** 2,)
        self.params_per_layer = self.parameters_shape[0] // DRCs

    def _conv_upload(self, params, img, kernel_size, stride, wires, wire=0):
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
        # number_of_kernel_uploads = 9 ## Error in calculate number_of_kernel_uploads
        params_per_upload = len(params) // number_of_kernel_uploads

        upload_counter = 0
        # wire = self.total_qbits - 1
        # print("HERE")
        # print(img.shape, kernel_size, stride)
        wire = -1
        for y in range(0, img.shape[1]-kernel_size+1, stride):
            for x in range(0, img.shape[0]-kernel_size+1, stride):
                # print(y, y+kernel_size, x, x+kernel_size)
                # print("imgae slide", img[y:y+kernel_size, x:x+kernel_size])
                # print( wires[wire])
                single_upload(params[upload_counter * params_per_upload: (upload_counter + 1) * params_per_upload],
                                   img[x:x+kernel_size, y:y+kernel_size], wires[wire])
                upload_counter = upload_counter + 1
                wire = wire - 1

    def encoder_1(self, params, inputs):
        """The encoder circuit for the SQAE

        Args:
            params (list): parameters to use
            inputs (list): inputs to upload
        """
        def circular_entanglement(wires):
            qml.CNOT(wires=[wires[-1] + 1, wires[0]])
            for i in wires:
                qml.CNOT(wires=[i, i+1])

        for i in range(self.DRCs):
            self._conv_upload(
                params[i * self.params_per_layer:(i + 1) * self.params_per_layer], 
                inputs, self.kernel_size, self.stride, 
                list(range(self.number_of_kernel_uploads)),
                wire=0
                )
            circular_entanglement(list(range(0, self.number_of_kernel_uploads - 1)))
            
    def encoder_2(self, params, inputs):
        """The encoder circuit for the SQAE

        Args:
            params (list): parameters to use
            inputs (list): inputs to upload
        """
        def circular_entanglement(wires):
            qml.CNOT(wires=[wires[0], wires[-1] + 1])
            for i in wires[::-1]:
                qml.CNOT(wires=[i+1, i])

        for i in range(self.DRCs):
            self._conv_upload(
                params[i * self.params_per_layer:(i + 1) * self.params_per_layer], 
                inputs, self.kernel_size, self.stride, 
                list(range(self.total_qbits - self.number_of_kernel_uploads, self.total_qbits)),
                wire=self.total_qbits-1
                )
            circular_entanglement(list(range(self.total_qbits - self.number_of_kernel_uploads, self.total_qbits - 1)))

    def circuit(self, params, inputs1, inputs2):
        """Full circuit to be used as SQAE

        includes encoder and SWAP test

        Args:
            params (list): parameters to be used for PQC
            inputs (list): inputs for the circuit

        Returns:
            expectation value of readout bit

        """
        # print("2 circuit", inputs.shape)
        self.encoder_1(params, inputs1)
        self.encoder_2(params, inputs2)

        # swap test
        qml.Hadamard(wires=self.total_qbits//2)
        for i in range(self.latent_qbits):
            qml.CSWAP(wires=[self.total_qbits//2, self.total_qbits//2 + i + 2, self.total_qbits//2 - i - 2])
        qml.Hadamard(wires=self.total_qbits//2)

        return qml.expval(qml.PauliZ(self.total_qbits//2))

    def plot_circuit(self):
        """Plots the circuit with dummy inputs
        """

        input2 = torch.rand(self.input_shape)
        input1 = torch.rand(self.input_shape)
        params = torch.rand(self.parameters_shape)
        fig, _ = qml.draw_mpl(self.circuit_node)(params, input1, input2)
        fig.show()



