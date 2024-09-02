import json


class Elements:
    def __init__(self, time=None, stress=None, thickness=None, temperature=None):
        self.Time = time
        self.Stress = stress
        self.Thickness = thickness
        self.Temperature = temperature


class OutgoingDto:
    def __init__(self, results: list[Elements] = None):
        self.Results = results if results is not None else []

    def append(self, elem: Elements):
        self.Results.append(elem)

    def num_elements(self) -> int:
        return len(self.Results)

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
