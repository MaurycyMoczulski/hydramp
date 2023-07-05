import numpy as np
import numpy.ma as ma  
from amp.inference.filtering import amino_based_filtering 

def select_peptides(results):
    peptides = np.array(results['pos_peptides']).reshape(64, -1).T
    amp = (results['pos_class_prediction'] < 0.8).reshape(64, -1)
    mic = results['pos_mic_prediction'].reshape(64, -1)
    combined = ma.masked_where(amp, mic)
    good = combined.argmax(axis=0)
    good_peptides = peptides[list(range(peptides.shape[0])), good]
    good_amp = np.array(results['pos_class_prediction']).reshape(64, -1).T[list(range(peptides.shape[0])), good]
    good_mic = np.array(results['pos_mic_prediction']).reshape(64, -1).T[list(range(peptides.shape[0])), good]
    return pd.DataFrame.from_dict({
        'sequence': good_peptides.tolist(), 
        'amp': good_amp.tolist(),
        'mic': good_mic.tolist(),
    })

def final_filtering(dataset):
    dataset = dataset[(dataset['amp'] > 0.8) & (dataset['mic'] > 0.8)]
    dataset = amino_based_filtering('../data/unlabelled_positive.csv', dataset)
    return dataset