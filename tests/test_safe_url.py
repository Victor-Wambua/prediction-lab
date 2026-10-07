from aviator_lab.collector import safe_url


def test_safe_url_strips_tokens():
    assert safe_url("https://airplane-next.spribegaming.com/?user=1&token=abc#x") == \
        "https://airplane-next.spribegaming.com/"
