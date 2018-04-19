import torch
from torch.autograd import Variable
from torch.nn import Module

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    r"""Function that measures Binary Cross Entropy between target and output
    logits.
    See :class:`~torch.nn.BCEWithLogitsLoss` for details.
    Args:
        input: Variable of arbitrary shape
        target: Variable of the same shape as input
        weight (Variable, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on size_average. When reduce
                is False, returns a loss per input/target element instead and ignores
                size_average. Default: True
    Examples::
         >>> input = torch.randn(3, requires_grad=True)
         >>> target = torch.FloatTensor(3).random_(2)
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss.backward()
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class BCEWithLogitsLoss(Module):
    r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.
    The loss can be described as:
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ t_n \cdot \log \sigma(x_n)
        + (1 - t_n) \cdot \log (1 - \sigma(x_n)) \right],
    where :math:`N` is the batch size. If reduce is ``True``, then
    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}
    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.
    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size
            "nbatch".
        size_average (bool, optional): By default, the losses are averaged
            over observations for each minibatch. However, if the field
            size_average is set to ``False``, the losses are instead summed for
            each minibatch. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on size_average. When reduce
            is False, returns a loss per input/target element instead and ignores
            size_average. Default: True
     Shape:
         - Input: :math:`(N, *)` where `*` means, any number of additional
           dimensions
         - Target: :math:`(N, *)`, same shape as the input
     Examples::
        >>> loss = nn.BCEWithLogitsLoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.FloatTensor(3).random_(2)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, weight=None, size_average=True, reduce=True):
        super(BCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        if self.weight is not None:
            var = Variable(self.weight) if not isinstance(self.weight, Variable) else self.weight
            return binary_cross_entropy_with_logits(input, target,
                                                    var,
                                                    self.size_average,
                                                    reduce=self.reduce)
        else:
            return binary_cross_entropy_with_logits(input, target,
                                                    size_average=self.size_average,
                                                    reduce=self.reduce)
