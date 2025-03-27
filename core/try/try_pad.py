import torch
from torch.nn.utils.rnn import pad_sequence


a = torch.randn(10, 2)
b = torch.randn(9, 2)
c = torch.randn(8, 2)

l = [(b, b, c, a), (a, a, c, b), (c, c, c, a)]
# d = pad_sequence([a, b, c], batch_first=True)
# print(d.shape, c.shape)


def pad_to_longest_each_element(batch):
    """
    batch: [(mic, ref, label), (...), ...]
    the input data, label must with shape (T,C) if time domain
    """
    # x[0] => mic => mic.shape[0] (T)
    # batch.sort(key=lambda x: x[0].shape[0], reverse=True)  # data length

    seq_len = [ele[0].size(0) for ele in batch]
    out = []
    mic, label, hl, _ = zip(*batch)  # B,T,C
    print("@1", mic[0].shape, mic[1].shape, len(mic))
    mic = pad_sequence(mic, batch_first=True).float()
    print("@", mic[0].shape, mic[1].shape, len(mic))
    for ele in zip(*batch):
        # print("1", ele[0].shape, ele[1].shape, ele[2].shape, len(ele))
        ele = pad_sequence(ele, batch_first=True).float()
        out.append(ele)
        # print(ele[0].shape, ele[1].shape)
    # mic = pad_sequence(mic, batch_first=True).float()
    # hl = pad_sequence(hl, batch_first=True).float()
    # label = pad_sequence(label, batch_first=True).float()

    # # data = pack_padded_sequence(data, seq_len, batch_first=True, enforce_sorted=True)
    # print(mic.shape)

    return *out, torch.tensor(seq_len)
    # return mic, label, hl, torch.tensor(seq_len)


a, b, c, d, N = pad_to_longest_each_element(l)
print(a[0].shape, N)
