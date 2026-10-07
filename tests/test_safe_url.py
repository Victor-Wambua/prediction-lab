from prediction_lab.collector import safe_url


def test_safe_url_strips_tokens():
    assert safe_url("https://game.example.com/?user=1&token=abc#x") == \
        "https://game.example.com/"
