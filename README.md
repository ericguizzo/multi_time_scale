This repo contains the pytorch implementation of Multi-Time-Scale convolution layer, as used in the paper "Multi-Time-Scale Convolution for Emotion Recognition from Speech Audio Signals", submitted for ICASSP 2020.

## USAGE
The layer can be used instead of a standard torch.nn.Conv2d layer inside a nn.Module class, providing the same parameters of a standard convolution. In addition, the following parameters can be specified:
* scale_factors: List of tuples with the desired scale factors for the 2 axes. Every tuple in the list creates a new parallel convolution branch inside the layer, which kernels are resampled with the corresponding factors. There is no limit on the amount of tuples that can be specified here.
* output_type: String. Possible values: 'pooled_map' (default) merge the parallel feature maps through 3d max pooling, 'concat_fmaps' concatenate parallel feature maps along the time dimension without pooling, 'interleave_chdim' interleave original and stretched kernels on a new axis. 'pooled_map' is the best in our experiments and avoids to augment the number of parameters.
* stretch_penality_lambda: float (0-1): penalty factor that penalizes the streched parallel branches, forcing the model to "prefer" the original feature map. 0 is the best in our experiments.
* training_mode: String. Possible values: 'train_and_eval' (default): use parallel branche both in training and evaluation, 'only_eval' use only original kernel for training and all parallel branches for evaluation, 'only_train' use only original kernel for evaluation and all parallel branches for training, 'only_gradients' always use the only original kernel, but update it according to the gradients of all parallel branches. 'train_and_eval' is the best in our experiments.


## EXAMPLE CLASS
```python
import torch
from torch import nn
import torch.nn.functional as F
from multiscale_convlayer2 import MultiscaleConv2d

class example(nn.Module):

    def __init__(self, layer_type, MTS_scale_factors, ,MTS_output_type, MTS_penalty):
        super(example, self).__init__()

        if layer_type == 'conv':
            self.conv = nn.Conv2d(1, 10, kernel_size=[10,5])
        elif layer_type == 'multi':
            self.conv = MultiscaleConv2d(1, 10, kernel_size=[10,5],
                        scale_factors=MTS_scale_factors, output_type=MTS_output_type,
                        stretch_penality_lambda=MTS_penalty)

        self.hidden = nn.Linear(6735, 200)
        self.out = nn.Linear(200, 8)


    def forward(self, X):
        training_state = self.training  #this serves for

        if self.layer_type == 'conv':
            X = F.relu(self.conv(X))
        if self.layer_type == 'multi':
            X = F.relu(self.conv(X, training_state))

        X = X.reshape(X.size(0), -1)
        X = F.relu(self.hidden(X))
        X = self.out(X)

        return X

```

## TRAINING
During the training you should average the MTS kernels in the end of the training loop, after the update of the variables. You can use a build-in function of MTS to do so. Just call the function update_kernels() on every MTS layer of a model. For example:
```python
for epoch in range(len(n_epochs)):
  #YOUR TRAINING CODE
  optimizer.step()
  for layer in model.modules():
      if isinstance(layer, MultiscaleConv2d):
          layer.update_kernels()
```

It is possible to compute the average usage of each stretch factor calling the function get_stretc_percs() on every MTS layer of a model. For example:
```python
stretch_percs = []
for layer in model.modules():
    if isinstance(layer, MultiscaleConv2d):
        temp_stretch_percs = layer.get_stretch_percs()
        stretch_percs.append(temp_stretch_percs)
```
