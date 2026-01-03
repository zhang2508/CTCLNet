import math
from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# 参考：https://github.com/alxndrTL/mamba.py/tree/main

def npo2(len):
    """
    Returns the next power of 2 above len
    """

    return 2 ** math.ceil(math.log2(len))


def pad_npo2(X):
    """
    Pads input length dim to the next power of 2

    Args:
        X : (B, L, D, N)

    Returns:
        Y : (B, npo2(L), D, N)
    """

    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)


class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        #  A : (B, D, L, N)
        #  X : (B, D, L, N)

        #  modifies X in place by doing a parallel scan.
        #  more formally, X will be populated by these values :
        #  H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        #  which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        #  only supports L that is a power of two (mainly for a clearer code)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        #  up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        #  we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        #  down sweep (first 2 steps unfolded)
        Aa = A[:, :, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
        Xa = X[:, :, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2 ** k - 1:L:2 ** k]
            Xa = X[:, :, 2 ** k - 1:L:2 ** k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        #  A : (B, D, L, N)
        #  X : (B, D, L, N)

        #  the same function as above, but in reverse
        # (if you flip the input, call pscan, then flip the output, you get what this function outputs)
        #  it is used in the backward pass

        #  only supports L that is a power of two (mainly for a clearer code)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        #  up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        #  we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        #  down sweep (first 2 steps unfolded)
        Aa = A[:, :, 0:L:2 ** (num_steps - 2)]
        Xa = X[:, :, 0:L:2 ** (num_steps - 2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 0:L:2 ** k]
            Xa = X[:, :, 0:L:2 ** k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.
        If you can, privilege sequence lengths that are powers of two.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """

        L = X_in.size(1)

        #  cloning is requiered because of the in-place ops
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            #  pad tensors (and clone btw)
            A = pad_npo2(A_in)  #  (B, npo2(L), D, N)
            X = pad_npo2(X_in)  #  (B, npo2(L), D, N)

        # prepare tensors
        A = A.transpose(2, 1)  #  (B, D, npo2(L), N)
        X = X.transpose(2, 1)  #  (B, D, npo2(L), N)

        #  parallel scan (modifies X in-place)
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        #  slice [:, :L] (cut if there was padding)
        return X.transpose(2, 1)[:, :L]

    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """

        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        # cloning is requiered because of the in-place ops
        if L == npo2(L):
            grad_output = grad_output_in.clone()
            #  the next padding will clone A_in
        else:
            grad_output = pad_npo2(grad_output_in)  #  (B, npo2(L), D, N)
            A_in = pad_npo2(A_in)  #  (B, npo2(L), D, N)

        # prepare tensors
        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1)  #  (B, D, npo2(L), N)
        A = torch.nn.functional.pad(A_in[:, :, 1:],
                                    (0, 0, 0, 1))  #  (B, D, npo2(L), N) shift 1 to the left (see hand derivation)

        #  reverse parallel scan (modifies grad_output in-place)
        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]


pscan = PScan.apply


@dataclass
class MambaConfig:
    d_model: int  #  D
    n_layers: int = 1
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16  #  N in paper/comments
    expand_factor: int = 2  #  E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  #  "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True

    pscan: bool = True  #  use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class _MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        #  projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        # old
        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                              kernel_size=config.d_conv, bias=config.conv_bias,
                              groups=config.d_inner,
                              padding=config.d_conv - 1)

        self.norm = RMSNorm(config.d_model)

        #  projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        #  projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        #  dt initialization
        #  dt weights
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))  #  inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        #  todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(config.d_inner))

        #  projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        #  x : (B, L, D)

        # y : (B, L, D)

        _, L, _ = x.shape
        skip = x
        x = self.norm(x)

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  #  (B, L, ED), (B, L, ED)

        #  x branch
        x = x.transpose(1, 2)  #  (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  #  depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  #  (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x)

        # z branch
        z = F.silu(z)

        output = y * z

        output = self.out_proj(output)
        output = output + skip
        return output

    def ssm(self, x):
        #  x : (B, L, ED)

        #  y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()
        #  T ODO remove .float()

        deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, L, ED)

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  #  (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y


#  taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


class Mamba(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand

        self.config = MambaConfig(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand_factor=self.expand_factor,
        )

        self.mamba = _MambaBlock(self.config)

    def forward(self, x):
        return self.mamba(x)

if __name__ == '__main__':
    batch, length, dim = 2, 64, 16
    x = torch.randn(batch, length, dim).to("cuda")
    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim,  # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,  # Local convolution width
        expand=2,  # Block expansion factor
    ).to("cuda")
    y = model(x)
    print(y.shape)