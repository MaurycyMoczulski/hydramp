import numpy as np
import keras
from itertools import cycle, islice

import numpy as np

class Ranker:
    def __init__(self, values):
        self.values = values
        self._recalculate()

    def _recalculate(self):
        self.rank = {}
        for it, val in enumerate(sorted(self.values)):
            self.rank[val] = it

    def add(self, new_values):
        self.values.extend(new_values)
        self._recalculate()

    def evaluate(self, value):
        return self.rank[value]


def label_value(amp_val, mic_val):
    return amp_val + mic_val


class ILSOGenerator(keras.utils.Sequence):
    def __init__(self, dataloader, amps, mics, output_names=None, alpha=1, enrichment_size=16):
        self.dataloader = dataloader
        values = [label_value(amp_val, mic_val) for amp_val, mic_val in zip(amps, mics)]
        self.data_len = len(values)
        self.ranker = Ranker(values)
        self.enrichment_size = enrichment_size
        self.alpha = alpha
        self.sequences = None
        self.amps = None
        self.mics = None
        self.output_names = output_names
        
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return 1500 # Infinite dataset in reality

    def add_examples(self, sequences, amps, mics):
        values = [label_value(amp_val, mic_val) for amp_val, mic_val in zip(amps, mics)]
        self.ranker.add(values)
        self.data_len += len(values)
        if self.sequences is not None:
            self.sequences = np.concatenate((self.sequences, sequences))
            self.amps = np.concatenate((self.amps, amps))
            self.mics = np.concatenate((self.mics, mics))
        else:
            self.sequences = sequences
            self.amps = amps
            self.mics = mics
        
    def __getitem__(self, index):
        inputs, outputs = next(self.dataloader)
        if self.sequences is not None:
            substitute = np.random.choice(len(inputs), self.enrichment_size)
            examples = np.random.choice(len(self.sequences), self.enrichment_size)
            for s, e in zip(substitute, examples):
                inputs[0][s] = self.sequences[e]
                inputs[1][s] = self.amps[e]
                inputs[2][s] = self.mics[e]
                outputs[0][s] = self.amps[e]
                outputs[1][s] = self.mics[e]
        if not self.output_names:
            return inputs, outputs
        return self.append_weights(inputs, outputs)
    
    def append_weights(self, inputs, outputs):
        amps = outputs[0]
        mics = outputs[1]
        vals = [label_value(amp_val, mic_val) for amp_val, mic_val in zip(amps, mics)]
        ranks = [self.ranker.evaluate(val) for val in vals]
        weights = np.array([1/(self.alpha * self.data_len + rank) for rank in ranks])
        
        return inputs, outputs, {k:weights for k in self.output_names}
