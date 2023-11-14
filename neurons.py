import time
from TimeSeries import TimeSeriesQueue
import random

from utils import NeuronInput

class Neuron():
    def __init__(self) -> None:
        self.init_time = time.time()
        self.spike_train = TimeSeriesQueue()
        self.last_update_time = self.init_time

    def update(self, input_data: NeuronInput):
        update_time = time.time()
        dt = update_time - self.last_update_time

        spike_value = self.spike_method(input_data, dt)

        self.spike_train.insert(spike_value)
        self.last_update_time = update_time
        return 0

    def spike_method(self, input_data: NeuronInput, dt):
        raise NotImplementedError("Spike method not implemented")

class SimpleNeuron(Neuron):
    def __init__(self) -> None:
        super().__init__()
    
    def spike_method(self, input_data: NeuronInput, dt):
        # Spikes in 20% of the cases
        return random.random() > 0.8
    ...

class HorizontalDirectionSimpleNeuron(Neuron):
    def __init__(self, direction:str) -> None:
        super().__init__()
        if direction=="right":
            self.direction = "right"
        else:
            self.direction = "left"
    
    def spike_method(self, input_data: NeuronInput, dt):
        # Spikes in 20% of the cases
        x_speed = input_data.mouse_speed[0]
        spike_chance = 0.1
        
        if self.direction == "right":
            if x_speed > 0:
                spike_chance = spike_chance * x_speed
        else:
            if x_speed < 0:
                spike_chance = spike_chance * x_speed * -1

        return spike_chance > random.random()
if __name__ == "__main__":
    # Usage
    neuron = SimpleNeuron()
    neuron.update(100)
    neuron.update(200)
    neuron.update(300)
    neuron.update(300)
    neuron.update(300)
    neuron.update(300)
    neuron.update(300)

    last_known_timestamp = 0
    updates = neuron.spike_train.fetch_updates(last_known_timestamp)
    print([u for u in updates if u[1]])  # Will print all updates since the epoch