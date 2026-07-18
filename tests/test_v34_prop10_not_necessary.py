from kappalogic import or_n_trigger_condition_is_not_necessary


def test_condition_is_neither_always_true_nor_always_false():
    # Prop10の条件(少なくとも1つの値が閾値超え)を意図的に満たさない
    # ようにサンプリングしても、good_match_rateもsevere_mismatch_rateも
    # 0や1にはならない(=Prop10の条件は必要条件ではないことの確認)
    r = or_n_trigger_condition_is_not_necessary(1e-4, n_trials=3000)
    assert 0.1 < r["good_match_rate"] < 1.0
    assert 0.0 < r["severe_mismatch_rate"] < 0.9


def test_good_match_rate_exceeds_severe_mismatch_rate():
    # 大半のケースでは(トリガーが無くても)実はうまくいく、という
    # 経験的な傾向を確認
    r = or_n_trigger_condition_is_not_necessary(1e-4, n_trials=3000)
    assert r["good_match_rate"] > r["severe_mismatch_rate"]


def test_result_keys_present():
    r = or_n_trigger_condition_is_not_necessary(1e-3, n_trials=500)
    assert "good_match_rate" in r
    assert "severe_mismatch_rate" in r
