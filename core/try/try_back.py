import torch
import torch.nn as nn


class A(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.register_parameter("weight", nn.Parameter(torch.ones(3, 2)))

    def forward(self, x):
        out = x @ self.weight
        return out

    def loss(self, x, y):
        return torch.mean((x - y) ** 2)


class B(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.register_parameter("weight", nn.Parameter(torch.ones(2, 2)))

    def forward(self, x):
        out = x @ self.weight
        return out

    def loss(self, x, y):
        return torch.mean((x - y) ** 2)


if __name__ == "__main__":
    inp = torch.ones(1, 3)
    lbl1 = torch.ones(1, 2)
    n1 = A()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, n1.parameters()))
    opt.zero_grad()
    out1 = n1(inp)
    l1 = n1.loss(out1, lbl1)
    print("1", n1.weight.grad)
    l1.backward()
    opt.step()
    print(n1.weight.grad, n1.weight)

    inp = torch.ones(1, 3)
    lbl1 = torch.ones(1, 2)
    lbl2 = torch.ones(1, 2) + 2.0
    n1 = A()
    n2 = B()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, n1.parameters()))
    opt.add_param_group({"params": filter(lambda p: p.requires_grad, n2.parameters())})
    opt.zero_grad()
    out1 = n1(inp)
    out2 = n2(out1)
    # l = n1.loss(out1, lbl1)
    l = n1.loss(out1, lbl1) + n2.loss(out2, lbl2)
    print("2", n1.weight.grad, n2.weight.grad)
    l.backward()
    opt.step()
    print("n1 grad", n1.weight.grad, n1.weight)
    print("n2 grad", n2.weight.grad, n2.weight)

    inp = torch.ones(1, 3)
    lbl1 = torch.ones(1, 2)
    lbl2 = torch.ones(1, 2) + 2.0
    n1 = A()
    n2 = B()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, n2.parameters()))
    opt.zero_grad()
    out1 = n1(inp)
    out2 = n2(out1)
    # l = n1.loss(out1, lbl1)
    l = n2.loss(out2, lbl2)
    print("3", n1.weight.grad, n2.weight.grad)
    l.backward()
    opt.step()
    print("n1 grad", n1.weight.grad, n1.weight)
    print("n2 grad", n2.weight.grad, n2.weight)
