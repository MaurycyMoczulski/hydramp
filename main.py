import tensorflow as tf
import amp.data_utils.data_loader as data_loader
import amp.data_utils.sequence as du_sequence
import numpy as np
import pandas as pd
from amp.config import hydra
from amp.models.decoders import amp_expanded_decoder
from amp.models.encoders import amp_expanded_encoder
from amp.models.master import master
from amp.utils import basic_model_serializer, callback, generator
from keras import backend, layers
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from amp.config import MIN_LENGTH, MAX_LENGTH, LATENT_DIM, MIN_KL, RCL_WEIGHT, HIDDEN_DIM, MAX_TEMPERATURE, K_DIM


config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8),
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

seed = 7
np.random.seed(seed)

kl_weight = backend.variable(MIN_KL, name="kl_weight")
tau = backend.variable(MAX_TEMPERATURE, name="temperature")

data_manager = data_loader.AMPDataManager(
    'data/unlabelled_positive.csv',
    'data/unlabelled_negative.csv',
    min_len=MIN_LENGTH,
    max_len=MAX_LENGTH)

amp_x, amp_y = data_manager.get_merged_data()
amp_x_train, amp_x_test, amp_y_train, amp_y_test = train_test_split(amp_x, amp_y, test_size=0.1, random_state=36)
amp_x_train, amp_x_val, amp_y_train, amp_y_val = train_test_split(amp_x_train, amp_y_train, test_size=0.2, random_state=36)

# Restrict the length
ecoli_df = pd.read_csv('data/mic_data.csv')
mask = (ecoli_df['sequence'].str.len() <= MAX_LENGTH) & (ecoli_df['sequence'].str.len() >= MIN_LENGTH)
ecoli_df = ecoli_df.loc[mask]
mic_x = du_sequence.pad(du_sequence.to_one_hot(ecoli_df['sequence']))
mic_y = ecoli_df.value

mic_x_train, mic_x_test, mic_y_train, mic_y_test = train_test_split(mic_x, mic_y, test_size=0.1, random_state=36)
mic_x_train, mic_x_val, mic_y_train, mic_y_val = train_test_split(mic_x_train, mic_y_train, test_size=0.2, random_state=36)

new_train = pd.DataFrame()
new_train['value'] = list(mic_y_train)
new_train['sequence'] = list(mic_x_train)
active_peptides = new_train[new_train.value < 1.5]
active_peptides = pd.concat([active_peptides] * 25)
new_train = pd.concat([
    new_train,
    active_peptides,
])
mic_x_train = np.array(list(new_train.sequence)).reshape(len(new_train), 25)
mic_y_train = np.array(new_train.value).reshape(len(new_train),)

bms = basic_model_serializer.BasicModelSerializer()
amp_classifier = bms.load_model('models/amp_classifier')
amp_classifier_model = amp_classifier()
mic_classifier = bms.load_model('models/mic_classifier/')
mic_classifier_model = mic_classifier()

encoder = amp_expanded_encoder.AMPEncoderFactory.get_default(HIDDEN_DIM, LATENT_DIM, MAX_LENGTH, K_DIM)
input_to_encoder = layers.Input(shape=(MAX_LENGTH,))
encoder_model = encoder(input_to_encoder)
decoder = amp_expanded_decoder.AMPDecoderFactory.build_default(LATENT_DIM, tau, MAX_LENGTH)
input_to_decoder = layers.Input(shape=(LATENT_DIM,))
decoder_model = decoder(input_to_decoder)

master_model = master.MasterAMPTrainer(
    amp_classifier=amp_classifier,
    mic_classifier=mic_classifier,
    encoder=encoder,
    decoder=decoder,
    kl_weight=kl_weight,
    rcl_weight=RCL_WEIGHT,
    k_dim=K_DIM,
    master_optimizer=Adam(lr=1e-3),
    loss_weights=hydra,
)

master_keras_model = master_model.build(input_shape=(MAX_LENGTH, 21))

master_keras_model.summary()

amp_amp_train = amp_classifier_model.predict(amp_x_train, verbose=1, batch_size=10000).reshape(len(amp_x_train))
amp_mic_train = mic_classifier_model.predict(amp_x_train, verbose=1, batch_size=10000).reshape(len(amp_x_train))
amp_amp_val = amp_classifier_model.predict(amp_x_val, verbose=1, batch_size=10000).reshape(len(amp_x_val))
amp_mic_val = mic_classifier_model.predict(amp_x_val, verbose=1, batch_size=10000).reshape(len(amp_x_val))

mic_amp_train = amp_classifier_model.predict(mic_x_train, verbose=1, batch_size=10000).reshape(len(mic_x_train))
mic_mic_train = mic_classifier_model.predict(mic_x_train, verbose=1, batch_size=10000).reshape(len(mic_x_train))
mic_amp_val = amp_classifier_model.predict(mic_x_val, verbose=1, batch_size=10000).reshape(len(mic_x_val))
mic_mic_val = mic_classifier_model.predict(mic_x_val, verbose=1, batch_size=10000).reshape(len(mic_x_val))

uniprot_x_train = np.array(du_sequence.pad(du_sequence.to_one_hot(pd.read_csv('data/Uniprot_0_25_train.csv').Sequence)))
uniprot_x_val = np.array(du_sequence.pad(du_sequence.to_one_hot(pd.read_csv('data/Uniprot_0_25_val.csv').Sequence)))

uniprot_amp_train = amp_classifier_model.predict(uniprot_x_train, verbose=1, batch_size=10000).reshape(len(uniprot_x_train))
uniprot_mic_train = mic_classifier_model.predict(uniprot_x_train, verbose=1, batch_size=10000).reshape(len(uniprot_x_train))
uniprot_amp_val = amp_classifier_model.predict(uniprot_x_val, verbose=1, batch_size=10000).reshape(len(uniprot_x_val))
uniprot_mic_val = mic_classifier_model.predict(uniprot_x_val, verbose=1, batch_size=10000).reshape(len(uniprot_x_val))

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

vae_callback = callback.VAECallback(
    encoder=encoder_model,
    decoder=decoder_model,
    tau=tau,
    kl_weight=kl_weight,
    amp_classifier=amp_classifier_model,
    mic_classifier=mic_classifier_model,
    output_layer=master_model.output_layer
)


sm_callback = callback.SaveModelCallback(
    model=master_model,
    model_save_path="models/final_models/",
    name="HydrAMP"
)

history = master_keras_model.fit_generator(
    training_generator,
    steps_per_epoch=10,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=5,
    callbacks=[vae_callback, sm_callback],
)
