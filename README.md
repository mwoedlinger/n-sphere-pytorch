# N-dimensional spherical coordinate transformations in pytorch
Efficient pytorch transformation function between cartesian and spherical coordinates in n-dimensions.

## Usage
Convert from cartesian coordinates to spherical coordinates:

```python
    >>> x_cartesian = torch.tensor([1.0, 1.0, 1.0])
    >>> x_spherical = to_spherical(x_cartesian)
    >>>  print(x_spherical)
    tensor([1.7321, 0.9553, 0.7854])
```

Convert from spherical to cartesian coordinates:

```python
    >>> x_cartesian = torch.tensor([1.7321, 0.9553, 0.7854])
    >>> x_cartesian = to_cartesian(x_spherical)
    >>>  print(x_spherical)
    tensor([1.0000, 1.0000, 1.0000])
```

The methods also support batched input. In general they expect a torch tensor of shape `(..., n)`.