import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,  # 320
        groups,  # 2
        vq_dim,  # 256
        # temp=(2, 0.5, 0.999995),
        temp=(1.5, 0.5, 0.995),
        combine_groups=False,
        activation=nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
        hard=True,
        std=0,
    ):
        """Vector quantization using gumbel softmax

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            groups: number of groups for vector quantization
            vq_dim: dimensionality of the resulting quantized vector
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            combine_groups: whether to use the vectors for all groups, num_group=1 if True.
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor

        Input: B,T,C(dim)
        CodeBook: 1, num_groups*num_vars, vq_dim//groups
        Output: B,T,vq_dim
        """
        super().__init__()
        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.hard = hard

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        # self.vars is the codebook initized by uniform sample
        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        if std == 0:
            nn.init.uniform_(self.vars)
        else:
            nn.init.normal_(self.vars, mean=0, std=std)

        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                nn.Linear(inner_dim, groups * num_vars),
            )
        else:  # True
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        if isinstance(temp, str):
            import ast

            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f"{temp}, {len(temp)}"
        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def get_codebook_indices(self):
        """cached the full index of codebook.

        :returns: (num_vars x ... x num_vars, group).flatten()

        """
        if self.codebook_indices is None:
            from itertools import product

            # [range(num_vars), range(num_vars)]
            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))  # num_vars x num_vars, groups
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            if not self.combine_groups:
                # num_var x num_var x ... num_var, group
                self.codebook_indices = self.codebook_indices.view(self.num_vars**self.groups, -1)
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def sample_from_codebook(self, b, n):
        """sample negative samples from codebook

        :param b: B,
        :param n: number of sample per batch
        :returns: B,N,-1

        """
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)  # num_var x num_var x .. x num_var
        cb_size = indices.size(0)

        assert n < cb_size, f"sample size {n} is greater than size of codebook {cb_size}"

        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))  # bxn,
        indices = indices[sample_idx]

        # self.vars: 1,Gxnum_var, var_dim
        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def set_num_updates(self, num_updates):
        self.curr_temp = max(self.max_temp * self.temp_decay**num_updates, self.min_temp)

    def diversity_loss(self, net_output):
        """compute the diversity loss

        (GV - \sum_{g=1}^G exp^{-\sum_{v=1}^V p_gv log(p_gv)} ) / GV

        :param net_output:
        :returns:

        """
        loss = (net_output["num_vars"] - net_output["prob_perplexity"]) / net_output["num_vars"]
        return loss

    def forward(self, x, produce_targets=False):
        """

        :param x:  B,T,C
        :param produce_targets:
        :returns: dict["x"] shape is B,T,vq_dim

        """
        result = {"num_vars": self.num_vars * self.groups}  # total codebook number

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)  # BT,C
        x = self.weight_proj(x)  # BT,(grops x num_vars)
        x = x.view(bsz * tsz * self.groups, -1)  # BTG,num_vars

        with torch.no_grad():
            # values, indices = x.max(-1)
            _, k = x.max(-1)  # k: BTG,
            hard_x = (
                x.new_zeros(*x.shape)  # x: BTG,num_vars
                .scatter_(-1, k.view(-1, 1), 1.0)  # fill 1 to the index of maximum values
                .view(bsz * tsz, self.groups, -1)
            )  # BT,G,num_vars
            hard_probs = torch.mean(hard_x.float(), dim=0)  # G,num_vars; p_{gv}
            result["code_perplexity"] = torch.exp(
                -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
            ).sum()  # scalar value

        # x: BT,G,num_vars(softmax) => G,num_vars
        avg_probs = torch.softmax(x.view(bsz * tsz, self.groups, -1).float(), dim=-1).mean(dim=0)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp

        if self.training:  # x: BTG,num_vars, is the input feature
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=self.hard).type_as(x)
        else:
            x = hard_x

        # NOTE: x is one-hot token
        x = x.view(bsz * tsz, -1)  # BT,Gxnum_vars

        vars = self.vars  # 1,Gxnum_vars,var_dim
        if self.combine_groups:  # false
            vars = vars.repeat(1, self.groups, 1)

        if produce_targets:  # false
            result["targets"] = (  # B,T,G
                x.view(bsz * tsz * self.groups, -1)  # BTG,num_vars
                .argmax(dim=-1)  # BTG
                .view(bsz, tsz, self.groups)
                .detach()
            )

        # x (BT, Gxnum_vars, 1) * (1, Gxnum_vars, var_dim)
        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.groups, self.num_vars, -1)  # BT, G, num_vars, vars_dim
        x = x.sum(-2)  # BT,G,vars_dim
        x = x.view(bsz, tsz, -1)  # BT, Gxvar_dim(vq_dim)

        result["x"] = x

        return result


if __name__ == "__main__":
    inp = torch.randn(1, 5, 4)
    net = GumbelVectorQuantizer(dim=4, num_vars=10, groups=2, vq_dim=6)
    out = net(inp, True)
    print(out["x"].shape, out["targets"].shape)

    idx = net.get_codebook_indices()
