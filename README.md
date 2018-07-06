# GAN-Hallucination
This projects investigates the possible hallucinations or adversarial attacks for solving linear inverse problems. The goal is to understand the possible hallucinations, define metrics to quantify the hallucination, and find regularization techniques to make deep reconstruction nets robust against hallucination.


# command to run 

python3  srez_main.py    
--run train     
--dataset_train /path/to/train/dataset      
--dataset_test /path/to/test/dataset    
--sampling_pattern  /path/to/sampling/pattern/mat/file     
--sample_size 256   
--sample_size_y 128    
--batch_size 2     
--summary_period  20000      
--sample_test -1   
--sample_train -1     
--subsample_test 1000   
--subsample_train 10000  
--train_time 3000   
--train_dir  /path/to/train/directory/save/results
--checkpoint_dir  /path/to/checkpoint/directory
--tensorboard_dir  /path/to/tensorboard
--gpu_memory_fraction 1.0  
--hybrid_disc 0    
--starting_batch 0
