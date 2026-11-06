import random

from sim_engine.hand_eval import best_rank, equity


def test_ranks_order():
    royal = best_rank(["As", "Ks", "Qs", "Js", "Ts"])
    wheel = best_rank(["Ah", "2d", "3c", "4s", "5h"])
    two_pair = best_rank(["Ah", "Ad", "Kc", "Kd", "2s", "2h", "9c"])
    assert royal[0] == 8
    assert wheel == (4, 5)
    assert two_pair == (2, 14, 13, 9)


def test_kicker_decides():
    ak = best_rank(["Ah", "Kd", "Ac", "7s", "4d", "3c", "2h"])
    a9 = best_rank(["Ah", "9d", "Ac", "7s", "4d", "3c", "2h"])
    assert ak > a9


def test_equity_reasonable():
    rng = random.Random(1)
    aa = equity(["As", "Ah"], [], iterations=1500, rng=rng)
    top_pair = equity(["7s", "9d"], ["8h", "4d", "9c"], iterations=800, rng=rng)
    k_high_river = equity(["4h", "Kc"], ["5d", "7c", "2c", "Jd", "6c"], iterations=800, rng=rng)
    assert 0.80 < aa < 0.90
    assert top_pair > 0.65
    assert k_high_river < 0.45


def test_vision_decision_folds_k_high_to_allin():
    from types import SimpleNamespace

    from sim_engine.vision_decision import decide

    state = SimpleNamespace(
        buttons={"fold": (135, 1469), "bet": (531, 1469)},
        button_text={"fold": "Fold", "bet": "All-in"},
        hero_cards=["4h", "Kc"],
        board_cards=["5d", "7c", "2c", "Jd", "6c"],
        pot=210320.0, hero_stack=69680.0, call_amount=0.0,
        enemy_stacks=[200000.0], dealer_box=None, stack_box=None,
    )
    choice = decide(state)
    assert choice is not None and choice.button == "fold"


def test_vision_decision_waits_without_check_call():
    from types import SimpleNamespace

    from sim_engine.vision_decision import decide

    state = SimpleNamespace(
        buttons={"fold": (135, 1469), "bet": (531, 1469)},
        button_text={"bet": "Bet"},
        hero_cards=["As", "Ah"], board_cards=[],
        pot=300.0, hero_stack=10000.0, call_amount=0.0,
        enemy_stacks=[], dealer_box=None, stack_box=None,
    )
    assert decide(state) is None
