import pandas as pd

from pearl.results import final_round_summary


def test_final_round_summary_uses_last_round_per_seed_and_method():
    df = pd.DataFrame(
        [
            {
                "seed": 1,
                "method": "a",
                "round": 0,
                "mean_global_accuracy": 0.1,
                "mean_global_macro_f1": 0.2,
                "worst_client_accuracy": 0.05,
                "neg_transfer_rate": 0.5,
                "selection_entropy": 0.7,
            },
            {
                "seed": 1,
                "method": "a",
                "round": 1,
                "mean_global_accuracy": 0.3,
                "mean_global_macro_f1": 0.4,
                "worst_client_accuracy": 0.2,
                "neg_transfer_rate": 0.1,
                "selection_entropy": 0.6,
            },
        ]
    )

    summary = final_round_summary(df)

    assert summary.loc[0, "method"] == "a"
    assert summary.loc[0, "mean_acc"] == 0.3
    assert summary.loc[0, "mean_f1"] == 0.4
