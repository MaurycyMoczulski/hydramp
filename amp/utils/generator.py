from itertools import cycle, islice

import numpy as np


def array_generator(array, batch_size):
    c = cycle(array)
    while True:
        yield np.array(list(islice(c, batch_size)))


def concatenated_generator(
        uniprot_x,
        uniprot_y,
        uniprot_mic,
        amp_x,
        amp_y,
        amp_mic,
        mic_x,
        mic_y,
        mic_mic,
        batch_size,
):
    amp_x_gen = array_generator(amp_x, batch_size)
    amp_y_gen = array_generator(amp_y, batch_size)
    amp_mic_gen = array_generator(amp_mic, batch_size)
    uniprot_x_gen = array_generator(uniprot_x, batch_size)
    uniprot_y_gen = array_generator(uniprot_y, batch_size)
    uniprot_mic_gen = array_generator(uniprot_mic, batch_size)
    mic_x_gen = array_generator(mic_x, batch_size)
    mic_y_gen = array_generator(mic_y, batch_size)
    mic_mic_gen = array_generator(mic_mic, batch_size)
    while True:
        batch_mic_x = next(mic_x_gen)
        batch_mic_y = next(mic_y_gen)
        batch_mic_mic = next(mic_mic_gen)
        batch_amp_x = next(amp_x_gen)
        batch_amp_y = next(amp_y_gen)
        batch_amp_mic = next(amp_mic_gen)
        batch_uniprot_x = next(uniprot_x_gen)
        batch_uniprot_y = next(uniprot_y_gen)
        batch_uniprot_mic = next(uniprot_mic_gen)
        result_x = np.concatenate([batch_amp_x, batch_mic_x, batch_uniprot_x])
        result_amp = np.concatenate([batch_amp_y, batch_mic_y, batch_uniprot_y])
        result_mic = np.concatenate([batch_amp_mic, batch_mic_mic, batch_uniprot_mic])
        noise_in = np.random.normal(0, 1.0, size=(result_amp.shape[0], 64))
        yield [
                  result_x,
                  result_amp,
                  result_mic,
                  noise_in
        ], \
              [
                  result_amp, # classifier output
                  result_mic, # regressor output
                  np.zeros_like(result_amp), # reconstruction
                  np.zeros_like(noise_in), # reconstructed error
                  np.zeros_like(noise_in), # unconstrained reconstructed error
              ]
