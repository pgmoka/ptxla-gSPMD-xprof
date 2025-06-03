# Sample of PyTorchXLA using gSPMD being analyzed with XProf

This repo contains a sample MNIST example using gSPMD with XProf.

To set-up your environment, follow the steps in https://github.com/pytorch/xla/blob/master/CONTRIBUTING.md.

You can then follow the steps in https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm to capture the profile generated, and analyze the model.

Note that this toy example is not particularly optimized, and aims primarily to serve as an example. In my case, I ran in a v6e-8 TPU.
