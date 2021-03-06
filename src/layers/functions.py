import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pyjet.backend as J
from pyjet.losses import bce_with_logits
from pyjet.backend import flatten

use_cuda = J.use_cuda


def fullpad1d(x, padding, value=0):
    return pad1d(x, (padding, padding), value=value)


def pad1d(x, padding, value=0):
    pad_l, pad_r = padding
    x = x.unsqueeze(3).contiguous()
    assert x.dim() == 4
    return F.pad(x, (0, 0, pad_l, pad_r), value=value).squeeze(3)


def softmax(input, axis=1):
    input_size = input.size()

    trans_input = input.transpose(axis, len(input_size) - 1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    soft_max_2d = F.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size) - 1)


def softkmax_mask(src_seq, x, k, kernel_size=3, use_cuda=use_cuda):
    exp_src = src_seq.expand(x.size(1), *src_seq.size()[1:])  # H x F x L
    exp_x = x.transpose(0, 1).expand(x.size(1), src_seq.size(1),
                                     *x.size()[2:])  # H x F x L-(ker-1)
    base_index = x.topk(k, dim=2)[1].sort(dim=2)[0].data  # 1 x H x k
    # H x F x k
    exp_index = base_index.transpose(0, 1).expand(base_index.size(1),
                                                  src_seq.size(1),
                                                  *base_index.size()[2:])
    mask_all_zeros = Variable(torch.zeros(*exp_x.size()))  # H x F x L-(ker-1)
    if use_cuda:
        mask_all_zeros = mask_all_zeros.cuda()
    # Create the softkmax_mask
    softmask = F.sigmoid(exp_x) * mask_all_zeros.scatter_(2, exp_index, 1.0)
    softmasked_src = Variable(torch.zeros(*exp_src.size()))
    if use_cuda:
        softmasked_src = softmasked_src.cuda()
    for i in range(kernel_size):
        # Run over the size of the kernel to capture the entire receptive
        # field
        softmasked_src += exp_src * \
            pad1d(softmask, (0 + i, kernel_size - 1 - i))
    # Average over the number of additions we did
    return softmasked_src / kernel_size


def kmax_mask(src_seq, x, k, kernel_size=3):
    exp_src = src_seq.expand(x.size(1), *src_seq.size()[1:])  # H x F x L
    base_index = x.topk(k, dim=2)[1].sort(dim=2)[0].data  # 1 x H x k
    # Initialize the mask and all ones mask for scattering
    all_zeros = Variable(torch.zeros(*exp_src.size()).cuda())
    for i in range(kernel_size):
        # Run over the size of the kernel to capture the entire receptive
        # field
        index = (base_index + i)
        # H x F x k
        exp_index = index.transpose(0, 1).expand(
            index.size(1), src_seq.size(1), *index.size()[2:])
        # H x F x L
        mask = mask.scatter_(2, exp_index, 1.0)
    return exp_src * mask


def kmax_pooling(x, dim, k):
    index = x.topk(min(x.size(dim), k), dim=dim)[1].sort(dim=dim)[0]
    x = x.gather(dim, index)
    if x.size(dim) < k:
        x = pad_tensor(x, k, dim=dim)
    return x


def kmax_select(x, att, k):
    # Dim should be 1 usually
    att_index = att.topk(k, dim=1)[1].sort(dim=1)[0]
    index = att_index.unsqueeze(1)
    index = index.expand(*x.size()[:2], index.size(-1))
    return x.gather(2, index), att_index


def pad_numpy_to_length(x, length):
    if len(x) < length:
        return np.concatenate([x, np.zeros((length - len(x),) + x.shape[1:])], axis=0)
    return x


def pad_tensor(tensor, length, pad_value=0.0, dim=0):
    # tensor is Li x E
    tensor = tensor.transpose(0, dim).contiguous()
    if tensor.size(0) == length:
        tensor = tensor
    elif tensor.size(0) > length:
        return tensor[:length]
    else:
        tensor = torch.cat([tensor, Variable(J.zeros(length - tensor.size(0), *tensor.size()[1:]).fill_(pad_value),
                                             requires_grad=False)])
    return tensor.transpose(0, dim).contiguous()


def pad_torch_embedded_sequences(tensors, pad_value=0.0, length_last=False):
    # tensors is B x Li x E
    # First find how long we need to pad until
    length_dim = -1 if length_last else 0
    assert len(tensors) > 0
    if length_last:
        assert all(tuple(seq.size())[:-1] == tuple(tensors[0].size())[:-1] for seq in tensors)
    else:
        assert all(tuple(seq.size())[1:] == tuple(tensors[0].size())[1:] for seq in tensors)
    seq_lens = [seq.size(length_dim) for seq in tensors]
    max_len = max(seq_lens)
    # Out is B x L x E
    # print([tuple(pad_tensor(tensors[i], max_len).size()) for i in range(len(tensors))])
    if length_last:
        return torch.stack(
            [pad_tensor(tensors[i].transpose(0, length_dim), max_len, pad_value=pad_value).transpose(0, length_dim)
             for i in range(len(tensors))]), seq_lens

    return torch.stack([pad_tensor(tensors[i], max_len, pad_value=pad_value) for i in range(len(tensors))]), seq_lens


def unpad_torch_embedded_sequences(padded_tensors, seq_lens, length_last=False):
    length_dim = -1 if length_last else 0
    if length_last:
        return [padded_tensor.transpose(0, length_dim)[:seq_len].transpose(0, length_dim) for padded_tensor, seq_len in
                zip(padded_tensors, seq_lens)]
    return [padded_tensor[:seq_len] for padded_tensor, seq_len in zip(padded_tensors, seq_lens)]


def pack_torch_embedded_sequences(tensors):
    # tensors is B x Li x E
    assert len(tensors) > 0
    assert all(seq.size(1) == tensors[0].size(1) for seq in tensors)
    seq_lens = [seq.size(0) for seq in tensors]
    return torch.cat(tensors), seq_lens


def unpack_torch_embedded_sequences(packed_tensors, seq_lens):
    # Find the start inds of all of the sequences
    seq_starts = [0 for _ in range(len(seq_lens))]
    seq_starts[1:] = [seq_starts[i-1] + seq_lens[i-1] for i in range(1, len(seq_starts))]
    # Unpack the tensors
    return [packed_tensors[seq_starts[i]:seq_starts[i] + seq_lens[i]] for i in range(len(seq_lens))]


def apply_pad_mask(x, pad_mask):
    x_len = x.size(-1)
    pad_mask_size = [pad_mask.size(0)] + [1] * (len(x.size()) - 2) + [x_len]
    # print(pad_mask.size())
    # print(pad_mask_size)
    # print(pad_mask.narrow(1, 0, x_len).size())
    return x * pad_mask.narrow(1, 0, x_len).contiguous().view(*pad_mask_size)


def multi_bce_with_logits(outputs, targets, size_average=True):
    flat_outputs = outputs.view(outputs.size(0) * outputs.size(1), 1)
    flat_targets = targets.unsqueeze(1).expand(
        *outputs.size()).contiguous().view(flat_outputs.size(0))
    return bce_with_logits(flat_outputs, flat_targets, size_average=size_average)


def multi_bce_with_logits_seq(outputs, targets, size_average=True):
    flat_outputs = torch.cat(outputs, dim=0).unsqueeze(1)  # B*Li x 1
    flat_targets = Variable(J.Tensor(np.zeros((flat_outputs.size(0),))))  # B*Li
    start = 0
    for i, output in enumerate(outputs):
        end = start + output.size(0)
        flat_targets[start:end] = targets[i].expand(output.size(0))
        start = end
    return bce_with_logits(flat_outputs, flat_targets, size_average=size_average)


def timedistributed_softmax(x, return_padded=False):
    # x comes in as B x Li x F, we compute the softmax over Li for each F
    # softmax = nn.Softmax2d()
    x, lens = pad_torch_embedded_sequences(x, pad_value=-float('inf'))  # B x L x F
    shape = tuple(x.size())
    assert len(shape) == 3
    x = F.softmax(x, dim=1)
    assert tuple(x.size()) == shape
    if return_padded:
        return x, lens
    # Un-pad the tensor and return
    return unpad_torch_embedded_sequences(x, lens)  # B x Li x F
