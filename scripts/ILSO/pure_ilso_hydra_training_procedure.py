#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8),
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# In[2]:


import amp.data_utils.data_loader as data_loader
import amp.data_utils.sequence as du_sequence
import numpy as np
import pandas as pd
import sklearn
from amp.config import hydra
from amp.models.decoders import amp_expanded_decoder
from amp.models.encoders import amp_expanded_encoder
from amp.models.master import master
from amp.utils import basic_model_serializer, callback, generator
from keras import backend, layers
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


# In[3]:


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)


# In[4]:


seed = 7
np.random.seed(seed)


# In[5]:


from amp.config import MIN_LENGTH, MAX_LENGTH, LATENT_DIM, MIN_KL, RCL_WEIGHT, HIDDEN_DIM, MAX_TEMPERATURE

kl_weight = backend.variable(MIN_KL, name="kl_weight")
tau = backend.variable(MAX_TEMPERATURE, name="temperature")


# # Import data

# In[6]:


data_manager = data_loader.AMPDataManager(
    '../../data/unlabelled_positive.csv',
    '../../data/unlabelled_negative.csv',
    min_len=MIN_LENGTH,
    max_len=MAX_LENGTH)

amp_x, amp_y = data_manager.get_merged_data()
amp_x_train, amp_x_test, amp_y_train, amp_y_test = train_test_split(amp_x, amp_y, test_size=0.1, random_state=36)
amp_x_train, amp_x_val, amp_y_train, amp_y_val = train_test_split(amp_x_train, amp_y_train, test_size=0.2, random_state=36)


# In[7]:


# Restrict the length
ecoli_df = pd.read_csv('../../data/mic_data.csv')
mask = (ecoli_df['sequence'].str.len() <= MAX_LENGTH) & (ecoli_df['sequence'].str.len() >= MIN_LENGTH)
ecoli_df = ecoli_df.loc[mask]
mic_x = du_sequence.pad(du_sequence.to_one_hot(ecoli_df['sequence']))
mic_y = ecoli_df.value


# In[8]:


mic_x_train, mic_x_test, mic_y_train, mic_y_test = train_test_split(mic_x, mic_y, test_size=0.1, random_state=36)
mic_x_train, mic_x_val, mic_y_train, mic_y_val = train_test_split(mic_x_train, mic_y_train, test_size=0.2, random_state=36)


# In[9]:


new_train = pd.DataFrame()
new_train['value'] = list(mic_y_train)
new_train['sequence'] = list(mic_x_train)
active_peptides = new_train[new_train.value < 1.5]
active_peptides  = pd.concat([active_peptides] * 25)
new_train  = pd.concat([
    new_train,
    active_peptides,    
])
mic_x_train = np.array(list(new_train.sequence)).reshape(len(new_train), 25)
mic_y_train = np.array(new_train.value).reshape(len(new_train),)


# # Import pretrained classifiers

# In[10]:


bms = basic_model_serializer.BasicModelSerializer()
amp_classifier = bms.load_model('../../models/amp_classifier')
amp_classifier_model = amp_classifier()
mic_classifier = bms.load_model('../../models/mic_classifier/')
mic_classifier_model = mic_classifier() 


# # Set up cVAE

# In[11]:


input_to_encoder = layers.Input(shape=(MAX_LENGTH,))
input_to_decoder = layers.Input(shape=(LATENT_DIM + 2,))

def reset_models():
    global encoder, encoder_model, decoder, decoder_model, master_model, master_keras_model
    
    encoder = amp_expanded_encoder.AMPEncoderFactory.get_default(HIDDEN_DIM, LATENT_DIM, MAX_LENGTH)
    encoder_model = encoder(input_to_encoder)
    decoder = amp_expanded_decoder.AMPDecoderFactory.build_default(LATENT_DIM + 2, tau, MAX_LENGTH)
    decoder_model = decoder(input_to_decoder)
    master_model = master.MasterAMPTrainer(
        amp_classifier=amp_classifier,
        mic_classifier=mic_classifier,
        encoder=encoder,
        decoder=decoder,
        kl_weight=kl_weight,
        rcl_weight=RCL_WEIGHT, 
        master_optimizer=Adam(lr=1e-3),
        loss_weights=hydra,
    )
    master_keras_model = master_model.build(input_shape=(MAX_LENGTH, 21))

def load_model(path):
    AMPMaster = bms.load_model(path)
    global encoder, encoder_model, decoder, decoder_model, master_model, master_keras_model
    encoder = AMPMaster.encoder
    encoder_model =  AMPMaster.encoder(input_to_encoder)
    decoder = AMPMaster.decoder
    AMPMaster.decoder.activation.temperature = tau
    decoder_model = AMPMaster.decoder(input_to_decoder)
    master_model = master.MasterAMPTrainer(
        amp_classifier=amp_classifier,
        mic_classifier=mic_classifier,
        encoder=encoder,
        decoder=decoder,
        kl_weight=kl_weight,
        rcl_weight=RCL_WEIGHT, 
        master_optimizer=Adam(lr=1e-3),
        loss_weights=hydra,
    )
    master_keras_model = master_model.build(input_shape=(MAX_LENGTH, 21))

# In[12]:

reset_models()
# load_model("../../models/ilso_models/HydrAMP_pure/10")

# In[13]:


#DATASET_AMP/MIC_TRAIN/VAL

amp_amp_train = amp_classifier_model.predict(amp_x_train, verbose=1, batch_size=10000).reshape(len(amp_x_train))
amp_mic_train = mic_classifier_model.predict(amp_x_train, verbose=1, batch_size=10000).reshape(len(amp_x_train))
amp_amp_val = amp_classifier_model.predict(amp_x_val, verbose=1, batch_size=10000).reshape(len(amp_x_val))
amp_mic_val = mic_classifier_model.predict(amp_x_val, verbose=1, batch_size=10000).reshape(len(amp_x_val))

mic_amp_train = amp_classifier_model.predict(mic_x_train, verbose=1, batch_size=10000).reshape(len(mic_x_train))
mic_mic_train = mic_classifier_model.predict(mic_x_train, verbose=1, batch_size=10000).reshape(len(mic_x_train))
mic_amp_val = amp_classifier_model.predict(mic_x_val, verbose=1, batch_size=10000).reshape(len(mic_x_val))
mic_mic_val = mic_classifier_model.predict(mic_x_val, verbose=1, batch_size=10000).reshape(len(mic_x_val))

uniprot_x_train = np.array(du_sequence.pad(du_sequence.to_one_hot(pd.read_csv('../../data/Uniprot_0_25_train.csv').Sequence)))
uniprot_x_val = np.array(du_sequence.pad(du_sequence.to_one_hot(pd.read_csv('../../data/Uniprot_0_25_val.csv').Sequence)))

uniprot_amp_train = amp_classifier_model.predict(uniprot_x_train, verbose=1, batch_size=10000).reshape(len(uniprot_x_train))
uniprot_mic_train = mic_classifier_model.predict(uniprot_x_train, verbose=1, batch_size=10000).reshape(len(uniprot_x_train))
uniprot_amp_val = amp_classifier_model.predict(uniprot_x_val, verbose=1, batch_size=10000).reshape(len(uniprot_x_val))
uniprot_mic_val = mic_classifier_model.predict(uniprot_x_val, verbose=1, batch_size=10000).reshape(len(uniprot_x_val))


# In[14]:


training_generator = generator.concatenated_generator(
    uniprot_x_train,
    uniprot_amp_train,
    uniprot_mic_train,
    amp_x_train,
    amp_amp_train,
    amp_mic_train,
    mic_x_train,
    mic_amp_train,
    mic_mic_train,
    128
)
validation_generator = generator.concatenated_generator(
    uniprot_x_val,
    uniprot_amp_val,
    uniprot_mic_val,
    amp_x_val,
    amp_amp_val,
    amp_mic_val,
    mic_x_val,
    mic_amp_val,
    mic_mic_val,
    128
)


# In[15]:


from amp.data_utils.ilso_dataloader import ILSOGenerator

aggregated_amp_train = np.concatenate((amp_amp_train, mic_amp_train, uniprot_amp_train))
aggregated_amp_val = np.concatenate((amp_amp_val, mic_amp_val, uniprot_amp_val))

aggregated_mic_train = np.concatenate((amp_mic_train, mic_mic_train, uniprot_mic_train))
aggregated_mic_val = np.concatenate((amp_mic_val, mic_mic_val, uniprot_mic_val))

training_dataset = ILSOGenerator(
    training_generator, aggregated_amp_train, aggregated_mic_train, 
    [
        "amp_prediction",
        "mic_prediction",
        "vae_loss_1",
        "mic_mean_grad",
        "amp_mean_grad",
        "unconstrained_sleep_mic_output_grad_input",
        "unconstrained_sleep_amp_output_grad_input",
        "correction_sleep_mic_output_grad",
        "correction_sleep_amp_output_grad",
        "correction_sleep_amp_prediction",
        "correction_sleep_mic_prediction",
        "unconstrained_sleep_amp_prediction",
        "unconstrained_sleep_mic_prediction",
        "z_cond_reconstructed_error",
        "correction_sleep_cond_reconstructed_error",
        "unconstrained_sleep_cond_reconstructed_error",
    ], enrichment_size=32)

val_dataset = ILSOGenerator(validation_generator, aggregated_amp_val, aggregated_mic_val) 


# In[16]:


from amp.utils.generate_peptides import get_unique, get_z_sigma, single_move_zeros
from amp.data_utils.sequence import pad, to_one_hot

peptides = [
    "ILRWPWWPWRRK", # Omiganan (MBI-226)
    "AKRHHGYKRKFH", # Demegen (P-113)
    "GRRRRSVQWCA", # hLF1-11
    "KNKRKRRRRRRGGRRRR", # Mel4
    "MPKEKVFLKIEKMGRNIRN", # Vishnu3 (HB-107)
]

n = 1000
temp = 2
batch_size=5000


def improve(seq, amp, mic, z_sigma, temp=0.0, batch_size=5000):
    z = encoder_model.predict(seq, batch_size=batch_size)
    noise = np.random.normal(loc=0, scale=temp*z_sigma, size=z.shape)
    encoded = z + noise
    conditioned = np.hstack([
        encoded,
        np.ones((len(seq), 1)),
        np.ones((len(seq), 1)),
    ])
    decoded = decoder_model.predict(conditioned, batch_size=batch_size)
    new_peptides = np.argmax(decoded, axis=2)
    new_peptides = np.array([single_move_zeros(y) for y in new_peptides])
    new_amp = amp_classifier_model.predict(new_peptides, batch_size=batch_size)
    new_mic = mic_classifier_model.predict(new_peptides, batch_size=batch_size)                                                                             
    
    # RELATIVE
    rel_better = new_amp > amp.reshape(-1, 1)
    rel_better = rel_better & (new_mic > mic.reshape(-1, 1))
    rel_better = np.logical_or.reduce(rel_better, axis=1)
    rel_improved = new_peptides[np.where(rel_better), :].reshape(-1, 25)
    before_rel_improve = seq[np.where(rel_better), :].reshape(-1, 25)
    
    # ABSOLUTE
    abs_better = new_amp >= 0.8
    abs_better = abs_better & (new_mic > 0.5)
    abs_better = np.logical_or.reduce(abs_better, axis=1)
    abs_improved = new_peptides[np.where(abs_better), :].reshape(-1, 25)
    before_abs_improve = seq[np.where(abs_better), :].reshape(-1, 25)
    
    return {
        'new_peptides': new_peptides,
        'rel_improved': rel_improved,
        'abs_improved': abs_improved,
        'before_rel_improve': before_rel_improve,
        'before_abs_improve': before_abs_improve, 
        'new_amp': new_amp,
        'new_mic': new_mic,
        }   

def get_new_peptides():
    analogues = []
    new_amps = []
    new_mics = []
    for peptide in peptides:
        pep = pad(to_one_hot([peptide]))
        amp = amp_classifier_model.predict(pep)
        mic = mic_classifier_model.predict(pep)
        z_sigma = get_z_sigma(encoder, pep)

        pep = np.vstack([pep] * n).reshape(-1, 25)
        amp = np.vstack([amp] * n)
        mic = np.vstack([mic] * n)

        analogue_batch = get_unique(improve(pep, amp, mic, z_sigma, temp=temp, batch_size=batch_size)['abs_improved'])
        new_amps.extend(amp_classifier_model.predict(analogue_batch, batch_size=5000))
        new_mics.extend(mic_classifier_model.predict(analogue_batch, batch_size=5000))
        analogues.extend(analogue_batch)
    return analogues, np.array(new_amps).flatten(), np.array(new_mics).flatten()


# In[17]:


import keras
class BestGenerationAdder(keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        seq, amps, mics = get_new_peptides()
        self.dataset.add_examples(seq, amps, mics)


# In[18]:


vae_callback = callback.VAECallback(
    encoder=encoder_model,
    decoder=decoder_model,
    tau=tau,
    kl_weight=kl_weight,
    amp_classifier=amp_classifier_model,
    mic_classifier=mic_classifier_model,
)

# In[19]:
add_callback = BestGenerationAdder(training_dataset)

for it in [1, 2]:
    print (f"Iteration {it}/2")
    
    sm_callback = callback.SaveModelCallback(
        model = master_model,
        model_save_path="../../models/ilso_models/",
        name=f"HydrAMP_restarting_phase_{it}")
    
    add_callback.on_epoch_end(None)
    reset_models()
    history = master_keras_model.fit_generator(
        training_dataset,
        steps_per_epoch=1408,
        epochs=11,
        validation_data=validation_generator,
        validation_steps=176,
        callbacks=[vae_callback, sm_callback],
    )


# In[ ]:




