import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# PyTorch/XLA specific imports
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
import torch_xla.distributed.spmd as xs
import os

import ray
from ray.train.backend import Backend, BackendConfig
from ray.train._internal.utils import get_address_and_port
from ray.train.torch.xla.config import TorchXLAConfig
from ray.train._internal.worker_group import WorkerGroup
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"


# Enable the SPMD
ray.init()
# xr.use_spmd()

# Declare mesh meshes
num_devices = xr.global_runtime_device_count()
device_ids = np.arange(num_devices)
conv_mesh_shape = (int(num_devices/2), 2, 1, 1)
conv_mesh = xs.Mesh(device_ids, conv_mesh_shape, ('data', 'dim1', 'dim2', 'dim3'))

linear_mesh_shape = (int(num_devices/2), 2)
linear_mesh = xs.Mesh(device_ids, linear_mesh_shape, ('data', 'model'))

# Define the CNN Model
class MNISTNet(nn.Module):
    def __init__(self):
      super(MNISTNet, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1).to(xm.xla_device())
      # xs.mark_sharding(self.conv1.weight, conv_mesh, ('data', None, None, None))

      self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1).to(xm.xla_device())
      # xs.mark_sharding(self.conv2.weight, conv_mesh, ('data', None, None, None))

      self.fc1 = nn.Linear(7*7*64, 128).to(xm.xla_device()) # Adjusted for 28x28 image, 2 pooling layers
      # xs.mark_sharding(self.fc1.weight, linear_mesh, ('data', None))

      self.fc2 = nn.Linear(128, 10).to(xm.xla_device())
      # xs.mark_sharding(self.fc2.weight, linear_mesh, ('data', 'model'))

    def forward(self, x):
      x = F.relu(F.max_pool2d(self.conv1(x), 2))
      x = F.relu(F.max_pool2d(self.conv2(x), 2))
      x = x.view(-1, 7*7*64) # Flatten the tensor
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return F.log_softmax(x, dim=1)

def log(txt):
    rank = os.environ.get("RANK", "unk")
    print(f"{rank}: {txt}", flush=True)

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
    mnist_data = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    # 2. Initialize the model and move it to the XLA device
    model = MNISTNet().to(device)
    model = ray.train.torch.prepare_model(model)

    # Define loss function and optimizer
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # 3. Move data and target to the XLA device
            data, target = data.to(device), target.to(device)

            # 4. Shard input
            # xs.mark_sharding(data, conv_mesh, ('data', 'dim1', None, None))

            output = model(data)

            loss = loss_fn(output, target)
            loss.backward()

            optimizer.step()

            torch_xla.sync()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    print("Training finished!")

if __name__ == '__main__':
  trainer = TorchTrainer(
      train_mnist,
      torch_config=TorchXLAConfig(),
      scaling_config=ScalingConfig(num_workers=1, use_gpu=False)
  )
  results = trainer.fit()
  print(results)
