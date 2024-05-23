from torch.autograd import Function


class GradReverse(Function):
    @ staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_output):
        return grad_output[0] * -ctx.lambd, None