import torch
import torch.nn.functional as F
from einops import rearrange

from comps.PMSQE.pmsqe_asteroid import SingleSrcPMSQE
from torch import Tensor
from typing import Callable

TYPE_LOSS_FN = Callable[..., torch.Tensor]


def loss_lsd_mag(sph_spec, est_spec):
    """Log Spectral Distance(LSD)
    Args:
        sph_spec: b,2,t,f
    """
    assert sph_spec.ndim == 4
    eps = torch.finfo(sph_spec.dtype).eps

    # if not torch.is_complex(sph_spec):
    #     sph_spec = rearrange(sph_spec, "b c t f -> b t f c").contiguous()
    #     sph_spec = torch.view_as_complex(sph_spec)

    # if not torch.is_complex(est_spec):
    #     est_spec = rearrange(est_spec, "b c t f -> b t f c").contiguous()
    #     est_spec = torch.view_as_complex(est_spec)  # B,T,F

    # torch.norm -> sum( abs(x)**p ) ** (1./p)
    mag_p = torch.norm(est_spec, p=2, dim=1)  # b,2,t,f->b,t,f
    mag_s = torch.norm(sph_spec, p=2, dim=1)

    sp = torch.log10(mag_p.square().clamp(eps))
    st = torch.log10(mag_s.square().clamp(eps))

    return (sp - st).square().mean(dim=-1).sqrt().mean()


def loss_pmsqe(sph_spec, est_spec, pad_mask=None):
    """Perceptally-Motivated Speech Quality
    Args:
        `sph_spec` and `est_spec`: features in frequency-domain
            with shape B,2,T,F
        pad_mask: indicate the padding frames with shape B, T, 1 where 1 for valid frame.
    """

    assert sph_spec.shape[-1] == 257

    def power(spec, freq_dim=-1):
        # B,T,2F -> ((B,T,F), (B,T,F))(chunk) -> B,T,F,2(stack) -> B,T,F(sum)
        return torch.stack(torch.chunk(spec, 2, dim=freq_dim), dim=-1).pow(2).sum(dim=-1)

    if pad_mask is None:
        pad_mask = torch.ones(est_spec.shape[0], est_spec.shape[2], 1, device=est_spec.device)

    # b,2,t,f -> b,t,2f
    sph_spec = rearrange(sph_spec, "b c t f->b t (c f)")
    est_spec = rearrange(est_spec, "b c t f->b t (c f)")

    power_sph_spec = power(sph_spec)
    power_est_spec = power(est_spec)

    # wD: Mean symmetric distortion.
    # wDA: Mean asymmetric distortion.
    # pmsq = PMSQE().cuda()
    # wD, wDA = pmsq(power_est_spec, power_sph_spec, pad_mask)
    # alpha = 0.1
    # score = alpha * (wD + 0.309 * wDA)

    pmsq = SingleSrcPMSQE(sample_rate=16000).cuda()
    score = pmsq(power_est_spec, power_sph_spec, pad_mask)
    score = score.mean()

    return score


def l2_norm(s: Tensor, keepdim=False):
    """
    sum(x^ord) ^ 1/ord
    """
    return torch.linalg.norm(s, dim=-1, keepdim=keepdim)


def loss_sisnr(sph: Tensor, est, zero_mean: bool = True) -> Tensor:
    """
    Args:
        sph: float tensor with shape `(..., time)`
        est: float tensor with shape `(..., time)`

    Returns:
        Float tensor with shape `(...,)` of SDR values per sample

    Example:
        >>> a = torch.tensor([1,2,3,4]).float()
        >>> b = torch.tensor([1,2,3,4]).float()
        >>> score = loss_sisnr(a, b)

    Algo:
        s_target = <sph, est> * sph / sph^2, where <> means inner dot
        e_noise = est - s_target
        sisnr = 10 * log_10(|s_target|^2 / |e_noise|^2)
    """
    # eps = torch.finfo(torch.float32).eps
    # eps = 1e-8
    eps = torch.finfo(sph.dtype).eps

    if zero_mean is True:
        s = sph - torch.mean(sph, dim=-1, keepdim=True)
        s_hat = est - torch.mean(est, dim=-1, keepdim=True)
    else:
        s = sph
        s_hat = est

    s_target = (
        (torch.sum(s_hat * s, dim=-1, keepdim=True) + eps)
        * s
        / (l2_norm(s, keepdim=True) ** 2 + eps)
    )
    e_noise = s_hat - s_target
    # sisnr_ = 10 * torch.log10((l2_norm(s_target) ** 2 + eps) / (l2_norm(e_noise) ** 2 + eps))
    sisnr = 10 * torch.log10(
        (torch.sum(s_target**2, dim=-1) + eps) / (torch.sum(e_noise**2, dim=-1) + eps)
    )
    return -torch.mean(sisnr)


def loss_seg_sisnr(sph, est, seg_n=1):
    """segment si-snr loss
    mic, est shape is (B, T)
    seg_n, is the number of segments
    """
    if sph.shape[1] % seg_n != 0:
        raise RuntimeError(
            f"length of audio data cannot be divided by segment length, {sph.shape[1] % seg_n}."
        )

    N = sph.shape[1] // seg_n

    sph = sph[:, None, None, :]
    est = est[:, None, None, :]

    seg_sph = F.unfold(sph, kernel_size=(1, N), stride=(1, N)).permute(0, 2, 1)
    seg_est = F.unfold(est, kernel_size=(1, N), stride=(1, N)).permute(0, 2, 1)

    seg_sph = seg_sph.reshape(-1, N)
    seg_est = seg_est.reshape(-1, N)

    loss = loss_sisnr(seg_sph, seg_est)
    return loss


def loss_snr(sph: Tensor, est: Tensor, zero_mean: bool = True) -> Tensor:
    r"""Calculate `Signal-to-noise ratio`_ (SNR_) meric for evaluating quality of audio.

    .. math::
        \text{SNR} = \frac{P_{signal}}{P_{noise}}

    where  :math:`P` denotes the power of each signal. The SNR metric compares the level of the desired signal to
    the level of background noise. Therefore, a high value of SNR means that the audio is clear.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        zero_mean: if to zero mean target and preds or not

    Returns:
        Float tensor with shape ``(...,)`` of SNR values per sample

    Raises:
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape

    Example:
        >>> from torchmetrics.functional.audio import signal_noise_ratio
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> signal_noise_ratio(preds, target)
        tensor(16.1805)
    """
    eps = torch.finfo(sph.dtype).eps
    if zero_mean:
        sph = sph - torch.mean(sph, dim=-1, keepdim=True)
        est = est - torch.mean(est, dim=-1, keepdim=True)

    noise = sph - est

    snr_value = 10 * torch.log10(
        (torch.sum(sph**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    )
    return -torch.mean(snr_value)


def loss_compressed_mag(sph: Tensor, est: Tensor, compress_factor=0.3):
    """
    Input:
        sph: specturm of sph, B,2,T,F
        est: specturm of sph, B,2,T,F

    Return: loss of mse_mag, mse_specs
    """
    mag_sph_2 = torch.maximum(
        torch.sum((sph * sph), dim=1, keepdim=True),
        torch.zeros_like(torch.sum((sph * sph), dim=1, keepdim=True)) + 1e-12,
    )
    mag_est_2 = torch.maximum(
        torch.sum((est * est), dim=1, keepdim=True),
        torch.zeros_like(torch.sum((est * est), dim=1, keepdim=True)) + 1e-12,
    )
    mag_sph_cpr = torch.pow(mag_sph_2, compress_factor / 2)  # B,1,T,F
    mag_est_cpr = torch.pow(mag_est_2, compress_factor / 2)  # B,1,T,F
    specs_sph_cpr = torch.pow(mag_sph_2, (compress_factor - 1) / 2) * sph  # B,2,T,F
    specs_est_cpr = torch.pow(mag_est_2, (compress_factor - 1) / 2) * est  # B,2,T,F

    mse_mag = torch.mean((mag_sph_cpr - mag_est_cpr) ** 2)
    mse_specs = torch.mean((specs_sph_cpr - specs_est_cpr) ** 2)
    return mse_mag, mse_specs


def anti_wrapping_fn(x):
    return torch.abs(x - torch.round(x / (2 * torch.pi)) * 2 * torch.pi)


def loss_phase(xk_sph, xk_est):
    """
    input with shape, b,2,t,f

    Paper: Ai, Y. and Ling, Z.H., 2023, June. Neural speech phase prediction based on parallel estimation architecture and anti-wrapping losses. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE.
    """
    sph_r, sph_i = xk_sph[:, 0, ...], xk_sph[:, 1, ...]  # b,t,f
    est_r, est_i = xk_est[:, 0, ...], xk_est[:, 1, ...]

    phase_r = torch.atan2(sph_i, sph_r)
    phase_g = torch.atan2(est_i, est_r)
    # instantaneous phase loss
    ip_loss = torch.mean(anti_wrapping_fn(phase_r - phase_g))
    # group delay loss
    gd_loss = torch.mean(anti_wrapping_fn(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))
    # instantaneous angular frequency
    iaf_loss = torch.mean(anti_wrapping_fn(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))

    phase_loss = ip_loss + gd_loss + iaf_loss
    return phase_loss, dict(
        ip_loss=ip_loss.detach(), gd_loss=gd_loss.detach(), iaf_loss=iaf_loss.detach()
    )


def loss_echo_aware(sph, est):
    """TODO unfinished
    Args:
        sph: float tensor with shape `(B, 2xF(r,i), T)` representing the F-domain feature.
        est: float tensor with shape `(B, 2xF, T)`

    Returns:
        Float tensor with shape `(...,)` of SDR values per sample

    Example:
        >>> a = torch.tensor([1,2,3,4]).float()
        >>> b = torch.tensor([1,2,3,4]).float()
        >>> score = loss_sisnr(a, b)
    """
    assert False
    sph_r, sph_i = torch.chunk(sph, 2, dim=1)
    est_r, est_i = torch.chunk(est, 2, dim=1)

    sph_mag = (sph_r**2 + sph_i**2) ** 0.5
    sph_mag_cpr = sph_mag**0.3
    est_mag = (est_r**2 + est_i**2) ** 0.5
    est_mag_cpr = est_mag**0.3

    sph_pha = torch.atan2(sph_i, sph_r)
    sph_cpr_r = sph_mag_cpr * torch.cos(sph_pha)
    sph_cpr_i = sph_mag_cpr * torch.sin(sph_pha)
    # cpr for compress power
    sph_cpr_specs = torch.concat([sph_cpr_r, sph_cpr_i], dim=1)

    est_pha = torch.atan2(est_i, est_r)
    est_cpr_r = est_mag_cpr * torch.cos(est_pha)
    est_cpr_i = est_mag_cpr * torch.sin(est_pha)
    est_cpr_specs = torch.concat([est_cpr_r, est_cpr_i], dim=1)

    loss_amp = (sph_mag_cpr - est_mag_cpr) ** 2

    # loss_amp = F.mse_loss(sph_mag_power_compress, est_mag_power_compress)
    # loss_pha = F.mse_loss(sph_cpr_specs, est_cpr_specs)


def loss_erle(mic: Tensor, est: Tensor):
    pow_mic = torch.mean(mic**2, dim=-1, keepdim=True)
    pow_est = torch.mean(est**2, dim=-1, keepdim=True)

    erle_score = 10 * torch.log10(pow_mic + torch.finfo(torch.float32).eps) - 10 * torch.log10(
        pow_est + torch.finfo(torch.float32).eps
    )
    erle_score = torch.minimum(erle_score, torch.tensor(30))
    return -erle_score


if __name__ == "__main__":
    import auraloss

    inp = torch.randn(1, 2, 10, 65)
    est = torch.randn(1, 2, 10, 65)
    # out = loss_lsd_mag(inp, est)

    inp = torch.randn(1, 16000)
    est = torch.randn(1, 16000)
    # print(out.shape, out)
    out = loss_sisnr(inp, est)
    out_ = loss_seg_sisnr(inp, est, 2)
    print(out.shape, out, out_)
    out = auraloss.time.SISDRLoss()(est, inp)
    print(out.shape, out)

    out = loss_snr(inp, est)
    print(out.shape, out)
    out = auraloss.time.SNRLoss()(est, inp)
    print(out.shape, out)
