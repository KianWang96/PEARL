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


def test_config_accepts_cifar10_and_dynamic_controls():
    cfg = config_from_mapping(
        {
            "dataset": "cifar10",
            "model_width": 32,
            "active_probability": 0.6,
            "descriptor_refresh_period": 10,
        }
    )

    assert cfg.dataset == "cifar10"
    assert cfg.model_width == 32
    assert cfg.active_probability == 0.6
    assert cfg.descriptor_refresh_period == 10


def test_config_rejects_invalid_dynamic_controls():
    for mapping in (
        {"active_probability": 0.0},
        {"active_probability": 1.1},
        {"descriptor_refresh_period": 0},
    ):
        try:
            config_from_mapping(mapping)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid config to fail: {mapping}")


def test_config_rejects_unknown_method_names():
    try:
        config_from_mapping({"methods": ["pearl_full", "typo_method"]})
    except ValueError as exc:
        assert "typo_method" in str(exc)
    else:
        raise AssertionError("Expected unknown methods to fail during config loading.")
