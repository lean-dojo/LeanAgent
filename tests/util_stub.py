def suggest_tactics(state, top_k=3):
    """Return a dummy tactic list (acts like a stand-in for the real generator)."""
    return ["rfl"]      # works for the goal ⊢ 1 = 1
