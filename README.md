# GAN-Hallucination
This project investigates the possible "hallucinations" that can be generated when solving linear inverse problems in the realm of medical imaging. The goal is to find such hallucinations, define metrics to quantify them, and identify regularization techniques to make deep neural net reconstructions robust against this sort of artifact creation.


# command to run 

python3  srez_main.py    
--run train     
--dataset_train /Data/Knee-highresolution-19cases/train   
--dataset_test /Data/Knee-highresolution-19cases/test
--sampling_pattern  /Data/Knee-highresolution-19cases/sampling_pattern/mask_2fold_160_128_knee_vdrad.mat     
--sample_size 160   
--sample_size_y 128    
--batch_size 2     
--summary_period  20000      
--sample_test -1   
--sample_train -1     
--subsample_test 1000   
--subsample_train 10000  
--train_time 3000   
--train_dir  /results
--checkpoint_dir  /checkpoints
--tensorboard_dir  /tensorboard
--gpu_memory_fraction 1.0  
--hybrid_disc 0    
--starting_batch 0
