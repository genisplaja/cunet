import logging
import numpy as np
import tensorflow as tf
from cunet.train.others.utilities import (
    make_earlystopping, make_reduce_lr, make_tensorboard, make_checkpoint,
    make_name, save_dir, write_config
)
from cunet.train.config import config
from cunet.train.models.cunet_model import cunet_model
from cunet.train.models.unet_model import unet_model
import os
import pdb

#from cunet.train.others.lock import get_lock

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.list_logical_devices('GPU')
print(gpus)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#tf.compat.v1.disable_eager_execution()

#log_dir = "./logs/fit/" + 'nov21'
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


logger = tf.get_logger()
logger.setLevel(logging.INFO)


def main():
    #tf.debugging.set_log_device_placement(True)


    config.parse_args()
    name = make_name()
    save_path = save_dir('models', name)
    write_config(save_path)
    #_ = get_lock()
    logger.info('Starting the computation')

    logger.info('Running training with config %s' % str(config))
    logger.info('Getting the model')
    if config.MODE == 'standard':
        model = unet_model()
    if config.MODE == 'conditioned':
        model = cunet_model()
    latest = tf.train.latest_checkpoint(
        os.path.join(save_path, 'checkpoint'))
    if latest:
        model.load_weights(latest)
        logger.info("Restored from {}".format(latest))
    else:
        logger.info("Initializing from scratch.")

    logger.info('Preparing the genrators')
    # Here to be sure that has the same config
    from cunet.train.data_loader import create_data_generator_train, create_data_generator_val

    ds_train = create_data_generator_train(config.BATCH_SIZE)
    ds_val = create_data_generator_val(config.BATCH_SIZE)

    logger.info('Starting training for %s' % name)

    # USE VAL_STEPS!! 

    with tf.device('/GPU:0'):
        model.fit(
            ds_train,
            validation_data=ds_val,
            steps_per_epoch=config.N_BATCH,
            epochs=config.N_EPOCH,
            verbose=1,
            validation_steps=config.N_BATCH//2,
            callbacks=[
                make_earlystopping(),
                make_reduce_lr(),
                make_tensorboard(save_path),
                make_checkpoint(save_path)
            ])

        logger.info('Saving model %s' % name)
        model.save(os.path.join(save_path, name+'.h5'))
        logger.info('Done!')
        return


if __name__ == '__main__':
    main()
