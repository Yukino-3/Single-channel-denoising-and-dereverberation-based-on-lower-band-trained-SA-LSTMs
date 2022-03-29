
Description of folders and files



config/
Lists of clean utterances for training and test.



DATA/
Mixtures, features, masks and separated speech are stored here.



dnn/
Code for DNN training and test, where dnn/main/ includes key functions for DNN training/test, dnn/pretraining/ includes code for unsupervised DNN pretraining.



gen_mixture/
Code for creating mixtures from noise and clean utterances.



get_feat/:
Code for acoustic features and ideal mask calculation.



premix_data/
Sample data including clean speech and factory noise.



load_config.m
Configures feature type, noise type, training utterance list, test utterance list, mixture SNR, mask type, etc.

RUN.m
Loads configurations from load_config.m and runs a speech separation demo.


This demo uses 600 mixtures for training and 120 mixtures for testing.
The mixtures are created by mixing clean utterances with factory noise at -2 dB.


A 4-hidden-layer DNN with sigmoid hidden activation is used for mask estimation.



To run this demo, simply execute RUN.m in matlab. 

This matlab script will execute the following steps:

    
I. Load configurations in load_config.m:
  
(a) 'train_list' and 'test_list' specify lists of clean utterances for training and test.
        
(b) 'mix_db' specifies the SNR of training and test mixtures.
      
(c) 'is_ratio_mask' specifies the mask type (0: binary mask, 1: ratio mask).
        (d) 'is_gen_mix', 'is_gen_feat' and 'is_dnn' indicate whether to perform different steps in speech separation.
    

II. Create data folders for this demo.
    

III. Implement DNN based speech separation. 
Three steps are performed:
       
(a) Generate training and test mixtures.
        
(b) Generate training and test features / masks.
        
(c) DNN training and test. To use a different network architecture, you may change the configurations ('opts.*') in ./dnn/main/dnn_train.m, where:
            
(1) 'opts.unit_type_hidden' specifies the activation function for hidden layers ('sigm': sigmoid, 'relu': ReLU).
            
(2) 'opts.isPretrain' indicates whether to perform pretraining (0: no pretraining, 1: pretraining). Note that pretraining is only supported for the sigmoid hidden activation function.
            
(3) 'opts.hid_struct' specifies the numbers of hidden layers and hidden units.
      (4) 'opts.sgd_max_epoch' specifies the maximum number of training epochs.
           (5) 'opts.isDropout' specifies whether to use dropout regularization.

When DNN training and test are finished, you will find the following speech separation results:
    DATA/factory/dnn/WAVE/db-2/: mixture, clean speech and resynthesized speech.
    DATA/factory/dnn/STORE/db-2/EST_MASK/: estimated masks and ideal masks.
    DATA/factory/log_db-2.txt: log file for this demo.


#######################
