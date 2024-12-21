- Norm flows can't can't work with discrete random variables, so we need to dequantize input image tensors.
Here the simplest solution [implemented](../src/modules/utils/tensors.py): adding a small amount of noise to each discrete value.
But in general it is better to use <a href="https://arxiv.org/abs/1902.00275">variational dequantization</a>.
- Read more about <a href="https://arxiv.org/abs/1605.08803v3">KL duality</a>.
