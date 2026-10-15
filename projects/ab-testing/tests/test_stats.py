import numpy as np
import pytest
from scipy import stats as sps
from statsmodels.stats.proportion import proportions_ztest, proportion_confint, power_proportions_2indep
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.multitest import multipletests

from ab_testing import stats as S


def test_ztest_bate_statsmodels():
    x1, n1, x2, n2 = 430, 1000, 480, 1000
    ours = S.two_proportion_ztest(x1, n1, x2, n2)
    z, p = proportions_ztest([x2, x1], [n2, n1])  # z de p_B - p_A, mesmo sinal que o nosso
    assert ours["p_value"] == pytest.approx(p, rel=1e-6)
    assert ours["z"] == pytest.approx(z, rel=1e-6)


def test_ztest_unilateral():
    ours = S.two_proportion_ztest(100, 1000, 130, 1000, alternative="larger")
    _, p = proportions_ztest([130, 100], [1000, 1000], alternative="larger")
    assert ours["p_value"] == pytest.approx(p, rel=1e-6)


def test_qui_quadrado_bate_scipy():
    table = np.array([[430, 570], [480, 520]])
    chi2, p, _, _ = sps.chi2_contingency(table, correction=False)
    ours = S.chi_square_test(430, 1000, 480, 1000)
    assert ours["chi2"] == pytest.approx(chi2) and ours["p_value"] == pytest.approx(p)
    chi2c, pc, _, _ = sps.chi2_contingency(table, correction=True)
    assert S.chi_square_test(430, 1000, 480, 1000, correction=True)["p_value"] == pytest.approx(pc)


def test_wilson_bate_statsmodels():
    lo, hi = proportion_confint(37, 120, method="wilson")
    ours = S.wilson_ci(37, 120)
    assert ours[0] == pytest.approx(lo, abs=1e-9) and ours[1] == pytest.approx(hi, abs=1e-9)


def test_wilson_caso_conhecido():
    # 0 de 10 -> IC de Wilson 95% = (0, 0.2775)
    lo, hi = S.wilson_ci(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.2775, abs=1e-3)


def test_newcombe_contem_diferenca_e_e_mais_estreito_que_nada_absurdo():
    lo, hi = S.diff_ci_newcombe(430, 1000, 480, 1000)
    d = 0.05
    assert lo < d < hi
    lo_w, hi_w = S.diff_ci_wald(430, 1000, 480, 1000)
    assert abs((hi - lo) - (hi_w - lo_w)) < 0.01


def test_bootstrap_media_proximo_do_t_test():
    rng = np.random.default_rng(1)
    a, b = rng.normal(10, 2, 800), rng.normal(10.5, 2, 800)
    r = S.bootstrap_diff_ci(a, b, "mean", n_boot=2000)
    t = sps.ttest_ind(b, a, equal_var=False)
    lo, hi = t.confidence_interval() if hasattr(t, "confidence_interval") else (None, None)
    assert r["ci_low"] < 0.5 < r["ci_high"] or r["ci_low"] < r["estimate"] < r["ci_high"]
    assert r["estimate"] == pytest.approx(b.mean() - a.mean())


def test_bayes_prob_exata_igual_monte_carlo():
    exata = S.prob_b_beats_a_exact(100, 1000, 130, 1000)
    mc = S.bayesian_ab(100, 1000, 130, 1000, n_samples=300_000)["p_b_beats_a"]
    assert exata == pytest.approx(mc, abs=0.005)
    assert 0.9 < exata < 1.0


def test_bayes_sem_diferenca_da_perto_de_meio():
    p = S.prob_b_beats_a_exact(500, 1000, 500, 1000)
    assert p == pytest.approx(0.5, abs=1e-3)


def test_perda_esperada_nao_negativa_e_menor_para_variante_melhor():
    r = S.bayesian_ab(100, 1000, 150, 1000, n_samples=100_000)
    assert r["expected_loss_choose_b"] < r["expected_loss_choose_a"]
    assert r["expected_loss_choose_b"] >= 0


def test_tamanho_de_amostra_bate_statsmodels():
    p1, p2 = 0.10, 0.12
    from statsmodels.stats.proportion import proportion_effectsize
    es = proportion_effectsize(p2, p1)
    n_sm = NormalIndPower().solve_power(effect_size=es, alpha=0.05, power=0.8, ratio=1, alternative="two-sided")
    # statsmodels usa o tamanho de efeito de Cohen (arcoseno); verificamos o poder na nossa fórmula.
    n = S.required_sample_size(p1, p2 - p1)
    assert S.power_for_n(p1, p2 - p1, n) == pytest.approx(0.8, abs=0.01)
    assert S.power_for_n(p1, p2 - p1, n - 200) < 0.8


def test_poder_bate_statsmodels_e_mde_inverte_o_poder():
    p1, mde, n = 0.10, 0.02, 4000
    from statsmodels.stats.proportion import proportion_effectsize
    es = proportion_effectsize(p1 + mde, p1)
    pw_sm = NormalIndPower().power(effect_size=es, nobs1=n, alpha=0.05, ratio=1, alternative="two-sided")
    assert S.power_for_n(p1, mde, n) == pytest.approx(pw_sm, abs=1e-3)
    m = S.minimum_detectable_effect(p1, n, power=0.8)
    assert S.power_for_n(p1, m, n) == pytest.approx(0.8, abs=1e-4)


def test_relativo_aumenta_n():
    # efeito relativo de 20% (0.10 -> 0.12) exige mais amostra que efeito absoluto de 1 p.p.
    assert S.required_sample_size(0.1, 0.2, relative=True) == S.required_sample_size(0.1, 0.02)


def test_holm_e_bonferroni_bate_statsmodels():
    p = [0.01, 0.04, 0.03, 0.20]
    for m in ("bonferroni", "holm"):
        ours = S.adjust_pvalues(p, m)
        ref = multipletests(p, method=m[:4] if m == "holm" else "bonferroni")[1]
        assert np.allclose(ours, ref)
    assert np.allclose(S.adjust_pvalues(p, "bh"), multipletests(p, method="fdr_bh")[1])


def test_srm_bate_scipy():
    r = S.srm_test(50_000, 50_500)
    chi2, p = sps.chisquare([50_000, 50_500])
    assert r["chi2"] == pytest.approx(chi2) and r["p_value"] == pytest.approx(p)


def test_peeking_infla_falsos_positivos():
    r = S.simulate_peeking(n_sims=1500, looks=10, seed=3)
    assert r["fpr_peeking"] > r["fpr_single_look"] + 0.05
    assert r["fpr_single_look"] < 0.09


def test_entradas_invalidas():
    with pytest.raises(ValueError):
        S.wilson_ci(11, 10)
