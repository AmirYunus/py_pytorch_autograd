import torch

# create a tensor and set `requires_grad = True` to track computation with it
x = torch.ones(2, 2, requires_grad = True)
print(f"x: {x}\n")

# do a tensor operation
y = x + 2
print(f"y: {y}\n")

# y was created as a result of an operation, so it has a `grad_fn`
print(f"y.grad_fn: {y.grad_fn}\n")

# do more operations on y
z = y * y * 3
out = z.mean()
print(f"z: {z}, out: {out}\n")

# `.requires_grad_(...) changes an existing tensor's `required_grad` flag in-place. the input flag defaults to `False` if not given
a = torch.randn(2,2)
a = ((a * 3) / (a - 1))
print(f"a.requires_grad: {a.requires_grad}\n")

a.requires_grad_(True)
print(f"a.requires_grad: {a.requires_grad}\n")

b = (a * a).sum()
print(f"b.grad_fn: {b.grad_fn}\n")

# let's backprop now because `out` contains a sinel scalar, `out.backward()` is equivalent to `out.backward(torch.tensor(1.))
out.backward()

# print gradients d(out)/dx
print(f"x.grad: {x.grad}\n")

# now let's take a look at an example of a vector-Hacobian product
x = torch.randn(3, requires_grad=True)
y = x * 2

while y.data.norm() < 1_000:
    y = y * 2

print(f"y: {y})\n")

# now in this case, y is no longer a scalar. `torch.autograd` could not compute the full Jacobian directly, but if we just want the vector-Jacobian product, simply pass the vector to `backward` as argument
v = torch.tensor([0.1, 1.0, 0.0001], dtype = torch.float)
y.backward(v)
print(f"x.grad: {x.grad}\n")

# you can also stop autograd from tracking history on tensors with `.requires_grad = True` wither by wrapping the code block in ` with torch.no_grad() or by using `.detach()` to get a new tensor with the same content but that does not require gradients

# `with torch_no_grad()`
print(f"x.requires_grad: {x.requires_grad}\n")
print(f"(x ** 2).requires_grad: {(x ** 2).requires_grad}\n")

with torch.no_grad():
    print(f"(x ** ).requires_grad: {(x ** 2).requires_grad}\n")

# `.detach()`
print(f"x.requires_grad: {x.requires_grad}\n")

y = x.detach()
print(f"y.requires_grad: {y.requires_grad}\n")
print(f"x.eq(y).all(): {x.eq(y).all()}\n")
