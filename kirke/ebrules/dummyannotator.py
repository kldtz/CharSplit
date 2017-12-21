
class DummyAnnotator:

    def __init__(self):
        pass

    # simply take everything
    def apply_rules(self, samples):
        return [1.0 for sample in samples]
