import pytest
from ab_testing import data as D


def test_sintetico_tem_schema_e_duas_versoes():
    df = D.validate(D.synthetic(n=2000))
    assert list(df.columns) == D.COLUMNS
    assert set(df["version"]) == {"gate_30", "gate_40"}


def test_validate_rejeita_schema_errado():
    df = D.synthetic(n=50).drop(columns=["retention_7"])
    with pytest.raises(ValueError):
        D.validate(df)
