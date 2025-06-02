import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# PyTorch/XLA specific imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
from torch_xla import runtime as xr
import torch_xla.distributed.spmd as xs
import os

# Enable the SPMD
xr.use_spmd()

# Declare mesh meshes
num_devices = xr.global_runtime_device_count()
device_ids = np.arange(num_devices)
conv_mesh_shape = (int(num_devices/2), 2, 1, 1)
conv_mesh = xs.Mesh(device_ids, conv_mesh_shape, ('data', 'dim1', 'dim2', 'dim3'))

linear_mesh_shape = (int(num_devices/2), 2)
linear_mesh = xs.Mesh(device_ids, linear_mesh_shape, ('data', 'model'))

os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

# Define the CNN Model
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1).to(xm.xla_device())
        xs.mark_sharding(self.conv1.weight, conv_mesh, ('data', None, None, None))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1).to(xm.xla_device())
        xs.mark_sharding(self.conv2.weight, conv_mesh, ('data', None, None, None))

        self.fc1 = nn.Linear(7*7*64, 128).to(xm.xla_device()) # Adjusted for 28x28 image, 2 pooling layers
        xs.mark_sharding(self.fc1.weight, linear_mesh, ('data', None))

        self.fc2 = nn.Linear(128, 10).to(xm.xla_device())
        xs.mark_sharding(self.fc2.weight, linear_mesh, ('data', 'model'))

    def forward(self, x):
        with xp.Trace('forward'):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 7*7*64) # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

def train_mnist():
    # Training parameters
    epochs = 5
    learning_rate = 0.01
    momentum = 0.5
    batch_size = 64

    # 1. Acquire the XLA device
    device = xm.xla_device()
    print(f"Running on XLA device: {device}")

    # Load MNIST dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    # 2. Initialize the model and move it to the XLA device
    model = MNISTNet().to(device)

    # Define loss function and optimizer
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            with xp.Trace('train_step_data_prep_and_forward'):
                optimizer.zero_grad()

                # 3. Move data and target to the XLA device
                data, target = data.to(device), target.to(device)

                # 4. Shard input
                xs.mark_sharding(data, conv_mesh, ('data', 'dim1', None, None))

                output = model(data)

            with xp.Trace('train_step_loss_and_backward'):
                loss = loss_fn(output, target)
                loss.backward()

            with xp.Trace('train_step_optimizer_step_host'):
                optimizer.step()

            torch_xla.sync()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    print("Training finished!")

if __name__ == '__main__':
    train_mnist()
