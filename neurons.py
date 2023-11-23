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

class StraightDirectionSimpleNeuron(SimpleNeuron):
    def __init__(self, direction:str) -> None:
        super().__init__()
        self.direction = direction
    
    def spike_method(self, input_data: NeuronInput, dt, baseline_noise=False):
        x_speed_factor = 1.5
        y_speed_factor = 2.5
        x_speed = input_data.mouse_speed[0] * x_speed_factor
        y_speed = input_data.mouse_speed[1] * y_speed_factor
        spike_chance = 0.03
        if self.direction == "left":
            speed = x_speed * -1
        elif self.direction == "right":
            speed = x_speed
        elif self.direction == "up":
            speed = y_speed * -1 
        elif self.direction == "down":
            speed = y_speed 
        else:
            raise ValueError("Invalid direction")
        if speed > 0:
            spike_chance = spike_chance * speed
        elif not baseline_noise:
            return 0
        return int(spike_chance > random.random())
    
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