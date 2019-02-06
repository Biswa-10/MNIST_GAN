from cGAN import cGAN
import tensorflow as tf
import os

if __name__ == '__main__':
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    tf.reset_default_graph()

    cgan = cGAN(discriminator_learning_rate=.0001, generator_learning_rate=.0004, epochs=100000,batch_size=500)

    '''   TRAIN  '''
    
    cgan.train()
            
    ''' TEST '''
    cgan.test()
