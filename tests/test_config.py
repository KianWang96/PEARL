from pearl.config import ExperimentConfig, apply_overrides, config_from_mapping


def test_config_overrides_coerce_types():
    cfg = apply_overrides(
        ExperimentConfig(),
        ["rounds=2", "lr=1e-4", "train_subset=null", "seeds=[4, 5]"],
    )

    assert cfg.rounds == 2
    assert cfg.lr == 1e-4
    assert cfg.train_subset is None
    assert cfg.seeds == [4, 5]


def test_config_rejects_unknown_keys():
    try:
        config_from_mapping({"missing": 1})
    except KeyError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected unknown config keys to raise KeyError.")
