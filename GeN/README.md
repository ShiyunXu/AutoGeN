### Arguments for Generalized Newton's method
Main function `lr_parabola` changes the learning rate based on pre-conditioned gradient, but does not change the model.

* `net (torch.nn.Module)`: The model to be optimized.
* `optimizer (torch.optim.Optimizer)`: The optimizer that pre-conditions the gradient.
* `criterion (callable, optional)`: Loss function. Default is `torch.nn.CrossEntropyLoss()`.
* `grad_accumulation_steps (int, optional)`: Number of gradient accumulation steps. Default is 1.
* `tr_iter (iterator, optional)`: Training data iterator for multiple forward passes on the same batch.
* `device (str, optional)`: Device to use (`'cuda'` or `'cpu'`). Default is `'cuda'`.
* `task (str, optional)`: Task type, one of `'image_cls'`, `'NLG'`, `'NLU'`. Customized task can be added by wrapping the forward pass.
* `scale (float, optional)`: Dependence on training horizon. Default is 1.0 (no dependence). Alternatively, set to a hyper-parameter-free decay from 1 to 0, e.g. 1-t/T.
* `lr_factor (list, optional)`: List of candidate learning rate multipliers. Default is `[-1,0,1]`.
* `verbose (bool, optional)`: Whether to print learning rates and losses. Default is False.

### Implementation
We use in-place operations to save memory and call `lr_parabola` infrequently (a.k.a. lazy update) to improve speed. This implementation should have minimal overhead compared to base optimizers. If you want to eliminate memory overhead, use CPU-offloading for `param.previous`.
