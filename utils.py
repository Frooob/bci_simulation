from dataclasses import dataclass

@dataclass
class NeuronInput:
    mouse_speed: tuple[int, int]
    ...
