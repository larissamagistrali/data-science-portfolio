"""Estatística de testes A/B: proporções, intervalos, bootstrap, bayesiano, poder e múltiplas métricas.

Convenção: A = controle, B = tratamento; x = conversões (sucessos), n = visitantes. O efeito é
sempre B - A (diferença absoluta de proporções), salvo onde indicado "relativo".
Tudo é testado contra scipy/statsmodels em tests/test_stats.py.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import integrate, optimize, stats

# ----------------------------------------------------------------------------- testes de hipótese


def _check_counts(x: int, n: int) -> None:
    if n <= 0 or x < 0 or x > n:
        raise ValueError(f"contagens inválidas: x={x}, n={n}")


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int, alternative: str = "two-sided") -> dict[str, float]:
    """Teste z de duas proporções (variância combinada sob H0: p1 = p2). Efeito = p2 - p1."""
    _check_counts(x1, n1)
    _check_counts(x2, n2)
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "larger":  # H1: p2 > p1
        p = stats.norm.sf(z)
    elif alternative == "smaller":  # H1: p2 < p1
        p = stats.norm.cdf(z)
    else:
        raise ValueError("alternative deve ser 'two-sided', 'larger' ou 'smaller'")
    return {"p1": p1, "p2": p2, "diff": p2 - p1, "z": float(z), "p_value": float(p), "se_pooled": float(se)}


def chi_square_test(x1: int, n1: int, x2: int, n2: int, correction: bool = False) -> dict[str, float]:
    """Qui-quadrado 2x2 de independência (grupo x conversão), com correção de Yates opcional."""
    _check_counts(x1, n1)
    _check_counts(x2, n2)
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]], dtype=float)
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    if (expected == 0).any():
        return {"chi2": 0.0, "p_value": 1.0, "dof": 1}
    diff = np.abs(table - expected)
    if correction:
        diff = np.maximum(diff - 0.5, 0)
    chi2 = float((diff**2 / expected).sum())
    return {"chi2": chi2, "p_value": float(stats.chi2.sf(chi2, 1)), "dof": 1}


def srm_test(n_a: int, n_b: int, expected_ratio: float = 0.5) -> dict[str, float]:
    """Sample Ratio Mismatch: o tamanho dos grupos é compatível com a divisão planejada? (qui-quadrado de aderência)"""
    total = n_a + n_b
    exp = np.array([total * expected_ratio, total * (1 - expected_ratio)])
    chi2 = float(((np.array([n_a, n_b]) - exp) ** 2 / exp).sum())
    return {"chi2": chi2, "p_value": float(stats.chi2.sf(chi2, 1)), "share_a": n_a / total}


# ----------------------------------------------------------------------------- intervalos de confiança


def wilson_ci(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo de Wilson para uma proporção."""
    _check_counts(x, n)
    z = stats.norm.ppf(1 - alpha / 2)
    p = x / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)


def diff_ci_wald(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> tuple[float, float]:
    """IC de Wald (variância não combinada) para p2 - p1."""
    p1, p2 = x1 / n1, x2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = stats.norm.ppf(1 - alpha / 2)
    d = p2 - p1
    return float(d - z * se), float(d + z * se)


def diff_ci_newcombe(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> tuple[float, float]:
    """IC de Newcombe (híbrido de Wilson) para p2 - p1: melhor cobertura que Wald com amostras pequenas."""
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson_ci(x1, n1, alpha)
    l2, u2 = wilson_ci(x2, n2, alpha)
    d = p2 - p1
    return float(d - np.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)), float(d + np.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2))


def bootstrap_diff_ci(
    a: np.ndarray, b: np.ndarray, statistic: str = "mean", n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """IC por bootstrap percentil para stat(b) - stat(a) (statistic: 'mean' ou 'median')."""
    func = {"mean": np.mean, "median": np.median}[statistic]
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = func(rng.choice(b, len(b))) - func(rng.choice(a, len(a)))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(func(b) - func(a)), "ci_low": float(lo), "ci_high": float(hi), "boot_std": float(diffs.std(ddof=1))}


def bootstrap_proportion_diff_ci(
    x1: int, n1: int, x2: int, n2: int, n_boot: int = 20000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """Bootstrap paramétrico-equivalente para proporções: reamostrar n Bernoulli iid equivale a Binomial(n, p-hat)."""
    rng = np.random.default_rng(seed)
    d = rng.binomial(n2, x2 / n2, n_boot) / n2 - rng.binomial(n1, x1 / n1, n_boot) / n1
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": x2 / n2 - x1 / n1, "ci_low": float(lo), "ci_high": float(hi)}


# ----------------------------------------------------------------------------- bayesiano (Beta-Binomial)


def beta_posteriors(x1: int, n1: int, x2: int, n2: int, prior: tuple[float, float] = (1.0, 1.0)):
    a0, b0 = prior
    return stats.beta(a0 + x1, b0 + n1 - x1), stats.beta(a0 + x2, b0 + n2 - x2)


def prob_b_beats_a_exact(x1: int, n1: int, x2: int, n2: int, prior: tuple[float, float] = (1.0, 1.0)) -> float:
    """P(theta_B > theta_A) por integração numérica: integral de f_B(x) * F_A(x) dx."""
    post_a, post_b = beta_posteriors(x1, n1, x2, n2, prior)
    lo, hi = min(post_a.ppf(1e-9), post_b.ppf(1e-9)), max(post_a.ppf(1 - 1e-9), post_b.ppf(1 - 1e-9))
    val, _ = integrate.quad(lambda t: post_b.pdf(t) * post_a.cdf(t), lo, hi, limit=200)
    return float(val)


def bayesian_ab(
    x1: int, n1: int, x2: int, n2: int, prior: tuple[float, float] = (1.0, 1.0), n_samples: int = 400_000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float]:
    """Beta-Binomial: P(B>A), perda esperada de escolher cada variante e IC credível do lift.

    Perda esperada de escolher B = E[max(theta_A - theta_B, 0)] (o quanto se perde, em média, se B
    for pior do que A); idem para A. Estimada por Monte Carlo (n_samples amostras da posterior).
    """
    post_a, post_b = beta_posteriors(x1, n1, x2, n2, prior)
    rng = np.random.default_rng(seed)
    ta, tb = post_a.rvs(n_samples, random_state=rng), post_b.rvs(n_samples, random_state=rng)
    diff = tb - ta
    lo, hi = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    rel = diff / ta
    rlo, rhi = np.percentile(rel, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "p_b_beats_a": float((tb > ta).mean()),
        "expected_loss_choose_b": float(np.maximum(ta - tb, 0).mean()),
        "expected_loss_choose_a": float(np.maximum(tb - ta, 0).mean()),
        "diff_mean": float(diff.mean()),
        "diff_ci_low": float(lo),
        "diff_ci_high": float(hi),
        "lift_rel_mean": float(rel.mean()),
        "lift_rel_ci_low": float(rlo),
        "lift_rel_ci_high": float(rhi),
    }


# ----------------------------------------------------------------------------- tamanho de amostra, MDE, poder


def _z(alpha: float, alternative: str) -> float:
    return float(stats.norm.ppf(1 - alpha / 2)) if alternative == "two-sided" else float(stats.norm.ppf(1 - alpha))


def required_sample_size(
    p_baseline: float, mde: float, alpha: float = 0.05, power: float = 0.8, alternative: str = "two-sided", relative: bool = False
) -> int:
    """Visitantes por grupo para detectar p2 = p1 + mde (ou p1*(1+mde) se relative=True).

    n = (z_a * sqrt(2 p_bar (1 - p_bar)) + z_b * sqrt(p1 (1 - p1) + p2 (1 - p2)))^2 / (p2 - p1)^2
    """
    p1 = p_baseline
    p2 = p1 * (1 + mde) if relative else p1 + mde
    if not (0 < p1 < 1 and 0 < p2 < 1) or p1 == p2:
        raise ValueError("p_baseline, p_baseline+mde devem estar em (0,1) e o efeito não pode ser zero")
    pbar = (p1 + p2) / 2
    za, zb = _z(alpha, alternative), float(stats.norm.ppf(power))
    n = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2
    return int(np.ceil(n))


def power_for_n(p_baseline: float, mde: float, n_per_group: float, alpha: float = 0.05, alternative: str = "two-sided", relative: bool = False) -> float:
    """Poder do teste (aproximação normal) para n por grupo, dado o efeito verdadeiro."""
    p1 = p_baseline
    p2 = p1 * (1 + mde) if relative else p1 + mde
    pbar = (p1 + p2) / 2
    za = _z(alpha, alternative)
    se0 = np.sqrt(2 * pbar * (1 - pbar) / n_per_group)
    se1 = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_group)
    d = abs(p2 - p1)
    return float(stats.norm.cdf((d - za * se0) / se1))


def minimum_detectable_effect(p_baseline: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8, alternative: str = "two-sided") -> float:
    """Menor efeito absoluto (positivo) detectável com n por grupo (inverte required_sample_size por busca de raiz)."""
    upper = min(1 - p_baseline - 1e-9, 1.0)

    def gap(mde: float) -> float:
        return power_for_n(p_baseline, mde, n_per_group, alpha, alternative) - power

    if gap(upper) < 0:
        return float("nan")  # nem um efeito máximo atinge o poder desejado
    return float(optimize.brentq(gap, 1e-9, upper, xtol=1e-10))


def power_curve(p_baseline: float, mde: float, ns: Sequence[int], alpha: float = 0.05, alternative: str = "two-sided", relative: bool = False) -> np.ndarray:
    return np.array([power_for_n(p_baseline, mde, n, alpha, alternative, relative) for n in ns])


# ----------------------------------------------------------------------------- múltiplas métricas


def adjust_pvalues(pvalues: Sequence[float], method: str = "holm") -> np.ndarray:
    """P-valores ajustados por Bonferroni, Holm (controlam FWER) ou Benjamini-Hochberg (controla FDR)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    order = np.argsort(p)
    adj = np.empty(m)
    if method == "holm":
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * p[idx])
            adj[idx] = min(running, 1.0)
        return adj
    if method in ("bh", "fdr_bh"):
        running = 1.0
        for rank in range(m - 1, -1, -1):
            idx = order[rank]
            running = min(running, p[idx] * m / (rank + 1))
            adj[idx] = running
        return adj
    raise ValueError("method deve ser 'bonferroni', 'holm' ou 'bh'")


# ----------------------------------------------------------------------------- "peeking"


def simulate_peeking(
    p: float = 0.1, n_per_group: int = 5000, looks: int = 10, alpha: float = 0.05, n_sims: int = 4000, seed: int = 0
) -> dict[str, float]:
    """Simula testes A/A (sem efeito real) e compara: olhar uma vez no fim x olhar `looks` vezes e parar no 1º p < alpha.

    Devolve a taxa de falsos positivos de cada estratégia. Com 1 olhada ~ alpha; com várias, bem maior.
    """
    rng = np.random.default_rng(seed)
    step = n_per_group // looks
    inc_a = rng.binomial(step, p, size=(n_sims, looks)).cumsum(axis=1)
    inc_b = rng.binomial(step, p, size=(n_sims, looks)).cumsum(axis=1)
    n = step * np.arange(1, looks + 1)
    pooled = (inc_a + inc_b) / (2 * n)
    se = np.sqrt(np.maximum(pooled * (1 - pooled) * 2 / n, 1e-18))
    z = (inc_b - inc_a) / n / se
    pv = 2 * stats.norm.sf(np.abs(z))
    return {
        "fpr_single_look": float((pv[:, -1] < alpha).mean()),
        "fpr_peeking": float((pv < alpha).any(axis=1).mean()),
        "looks": looks,
        "n_sims": n_sims,
        "alpha": alpha,
    }
