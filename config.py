import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

import os

flags = tf.app.flags


flags.DEFINE_string('ImageFileType', '.png', 'The file type of images')
flags.DEFINE_string('LabelFileType', '.tru', 'The extension of the file holding the ground-truth labels')


flags.DEFINE_integer('TRAIN_NB', 20, 'Number of training images to process')
flags.DEFINE_string('TRAIN_LIST', './samples/list', 'List of training data without file extension.')
flags.DEFINE_string('TRAIN_LOCATION', './samples/Images/', 'Location of training data. Could be included in the data list.')
flags.DEFINE_string('TRAIN_TRANS', './samples/Labels/', 'Location of training data transcriptions')


flags.DEFINE_integer('VAL_NB', 20, 'Number of validation images to process')
flags.DEFINE_string('VAL_LIST', './samples/list',	'List of validation data without file extension.')
flags.DEFINE_string('VAL_LOCATION', './samples/Images/', 'Location of validation data. Could be included in the data list.')
flags.DEFINE_string('VAL_TRANS', './samples/Labels/', 'Location of validation data transcriptions')


flags.DEFINE_integer('TEST_NB', 20, 'Number of test images to process')
flags.DEFINE_string('TEST_LIST', './samples/list', 'List of test data without file extension.')
flags.DEFINE_string('TEST_LOCATION', './samples/Images/', 'Location of test data. Could be included in the data list.')
flags.DEFINE_boolean('WriteDecodedToFile', True, 'Write the decoded text to file or stdout?')


flags.DEFINE_string('CHAR_LIST', './samples/CHAR_LIST', 'Sorted list of classes/characters. First one must be <SPACE>')


flags.DEFINE_string('SaveDir', './model', 'Directory where model checkpoints are saved')
flags.DEFINE_string('ModelName', 'model.ckpt', 'Name of the model checkpoints')
flags.DEFINE_string('LogFile', './log', 'Log file')
flags.DEFINE_string('LogDir', './summary', 'Directory to store Tensorflow summary information')
flags.DEFINE_string('Probs', './Probs', 'Directory to store posteriors for WFST decoder')


flags.DEFINE_boolean('LeakyReLU', True, 'Use Leaky ReLU or ReLU')


flags.DEFINE_integer('NUnits', 256, 'Number of LSTM units per forward/backward layer')
flags.DEFINE_integer('NLayers', 3, 'Number of BLSTM layers')


flags.DEFINE_integer('StartingEpoch', 0, 'The epoch number to start training from') # = 0 to train from scratch, != 0 to resume from the latest checkpoint
flags.DEFINE_float('LearningRate', 0.0005, 'Learning rate')
flags.DEFINE_integer('BatchSize', 10, 'Batch size') #This is actually the number of images to process each iteration
flags.DEFINE_boolean('RandomBatches', True, 'Randomize the order of batches each epoch')
flags.DEFINE_integer('MaxGradientNorm', 5, 'Maximum gradient norm')
flags.DEFINE_integer('SaveEachNEpochs', 1, 'Save model each n epochs')
flags.DEFINE_integer('NEpochs', 1000000, 'Run the training for n epochs')
flags.DEFINE_integer('TrainThreshold', 20, 'Stop the training after n epochs with no improvement on validation')

cfg = flags.FLAGS

if (os.path.exists(cfg.SaveDir) == False): os.makedirs(cfg.SaveDir)
if (os.path.exists(cfg.LogDir) == False): os.makedirs(cfg.LogDir)
