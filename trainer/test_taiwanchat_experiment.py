from taiwanchat_experiment import apply_terms, approved_policy, leading_letter


def test_self_containing_replacement_does_not_rewrite_correct_text():
    policy = approved_policy({"terms": {"算法": "演算法", "數據": "資料"}, "exclusions": []})
    assert apply_terms("這個演算法用數據", policy)[0] == "這個演算法用資料"
    assert apply_terms("這個算法", policy)[0] == "這個演算法"


def test_leading_letter_accepts_echoed_option_only():
    assert leading_letter("D. 約 70 萬次") == "D"
    assert leading_letter("B") == "B"
    assert leading_letter("Another answer") is None
