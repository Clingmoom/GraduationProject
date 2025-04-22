class SpecialToken:
    text = ""
    index_in_all_tokens = 0
    steps = []
    weight = 1.0
    def __init__(self, text, index_in_all_tokens, steps, weight) -> None:
        self.text = text
        self.index_in_all_tokens = index_in_all_tokens
        self.steps = steps
        self.weight = weight

    def __repr__(self) -> str:
        return f"Text:{self.text}\tIndex of all:{self.index_in_all_tokens}\tSteps:{self.steps}\tWeight:{self.weight}\n"