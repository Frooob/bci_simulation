import random

from neurons import HorizontalDirectionSimpleNeuron, Neuron, SimpleNeuron

class Brain():
    def __init__(self) -> None:
        self.neurons: list[Neuron] = []

    def get_recent_spikes(self, last_timestamp):
        recent_spikes = []
        for neuron in self.neurons:
            recent_spikes.append([msg[1] for msg in neuron.spike_train.fetch_updates(last_timestamp)])
        return recent_spikes

    def get_n_recent_spikes(self, n):
        recent_spikes = []
        for neuron in self.neurons:
            recent_spikes.append([msg[1] for msg in neuron.spike_train.data[-n:]])
        return recent_spikes
    
    def update(self, input_data):
        for neuron in self.neurons:
            neuron.update(input_data)
        

class StupidBrain(Brain):
    def __init__(self) -> None:
        super().__init__()
        self.neurons = [SimpleNeuron() for i in range(2)]
        self.update_rate = 0
    
    def update(self, update_data):
        """ only up
        """
        mod = 4
        if self.update_rate % mod == 0:
            super().update(update_data)
        self.update_rate += 1


class HorizontalDirectionBrain(StupidBrain):
    def __init__(self) -> None:
        super().__init__()
        self.neurons = [HorizontalDirectionSimpleNeuron("right"), HorizontalDirectionSimpleNeuron("left")]
