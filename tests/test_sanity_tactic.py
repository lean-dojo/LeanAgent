from tests.util_stub import suggest_tactics

def test_tactic_generator_nonempty():
    """Generator should emit at least one tactic string."""
    tactics = suggest_tactics("⊢ 1 = 1")
    assert tactics and any(t.strip() for t in tactics), "generator returned nothing"
