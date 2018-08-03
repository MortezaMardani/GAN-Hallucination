import numpy as np
import os.path
import scipy.misc
import tensorflow as tf
import time
import json
from scipy.io import savemat

from tensorflow.contrib.tensorboard.plugins import projector


FLAGS = tf.app.flags.FLAGS
# FLAGS.sample_size_y = FLAGS.sample_size if FLAGS.sample_size_y<0
OUTPUT_TRAIN_SAMPLES = 0

def compute_SNR(gt, recon):
    first_term =  np.linalg.norm(gt)
    #print("Norm of Ground Truth is: %f" % first_term)
    second_term = np.linalg.norm((gt - recon))
    #result = 20 * np.log10(first_term/second_term)
    result = -20 * np.log10(second_term/first_term)
    return result

def _summarize_progress(sess,feed_dict, sum_writer,train_data, feature, label, gene_output, gene_output_list, eta, nmse, kappa, batch, suffix, max_samples=2, gene_param=None):

    td = train_data

    size = [label.shape[1], label.shape[2]]
    print(size)

    # complex input zpad into r and channel
    complex_zpad = tf.image.resize_nearest_neighbor(feature, size)
    complex_zpad = tf.maximum(tf.minimum(complex_zpad, 1.0), 0.0)

    # zpad magnitude
    mag_zpad = tf.sqrt(complex_zpad[:,:,:,0]**2+complex_zpad[:,:,:,1]**2)
    mag_zpad = tf.maximum(tf.minimum(mag_zpad, 1.0), 0.0)
    mag_zpad = tf.reshape(mag_zpad, [FLAGS.batch_size,size[0],size[1],1])
    mag_zpad = tf.concat(axis=3, values=[mag_zpad, mag_zpad])
    
    # output image
    gene_output_complex = tf.complex(gene_output[:,:,:,0],gene_output[:,:,:,1])
    mag_output = tf.maximum(tf.minimum(tf.abs(gene_output_complex), 1.0), 0.0)
    mag_output = tf.reshape(mag_output, [FLAGS.batch_size, size[0], size[1], 1])
    #print('size_mag_output', mag)
    mag_output = tf.concat(axis=3, values=[mag_output, mag_output])




    label_complex = tf.complex(label[:,:,:,0], label[:,:,:,1])
    label_mag = tf.abs(label_complex)
    label_mag = tf.reshape(label_mag, [FLAGS.batch_size, size[0], size[1], 1])
    mag_gt = tf.concat(axis=3, values=[label_mag, label_mag])

    # concate for visualize image
    image = tf.concat(axis=2, values=[complex_zpad, mag_zpad, mag_output, mag_gt])
    image = image[0:FLAGS.batch_size,:,:,:]
    image = tf.concat(axis=0, values=[image[i,:,:,:] for i in range(int(FLAGS.batch_size))])
    image = td.sess.run(image)
    print('save to image size {0} type {1}', image.shape, type(image))

    # save magnitude of output images
    filename = 'magnitude_batch%06d_%s.out' % (batch, suffix)
    filename = os.path.join('mags',filename)
    np.save(filename,image)

    
    # 3rd channel for visualization
    mag_3rd = np.maximum(image[:,:,0],image[:,:,1])
    image = np.concatenate((image, mag_3rd[:,:,np.newaxis]),axis=2)

    # save to image file
    print('save to image,', image.shape)
    filename = 'batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(FLAGS.train_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))



    # Add the batch dimension
    dim1, dim2, dim3 = image.shape
    
    point1 = dim1 // 4
    point2 = dim2 // 2
    point3 = 3*(dim2 // 4)

    recon_img = image[:,point2:point3,:]
    real_img = image[:,point3:,:]
    SNR_batch_1 = compute_SNR(real_img,recon_img)
    print("Calculated SNR is:")
    print(SNR_batch_1)




    recon_img1 = image[:point1,point2:point3,:]
    real_img1 = image[:point1,point3:,:]

    SNR1 = compute_SNR(real_img1, recon_img1)
    print("Calculated SNR 1 is:")
    print(SNR1)
    print(real_img1.shape)
    print(recon_img1.shape)

    point_img2 = dim1 // 2

    recon_img2 = image[point1:point_img2,point2:point3,:]
    real_img2 = image[point1:point_img2,point3:,:]

    SNR2 = compute_SNR(real_img2, recon_img2)
    print("Calculated SNR 2 is:")
    print(SNR2)

    point_img3 = 3*(dim1 // 4)

    recon_img3 = image[point_img2:point_img3,point2:point3,:]
    real_img3 = image[point_img2:point_img3,point3:,:]

    SNR3 = compute_SNR(real_img3, recon_img3)
    print("Calculated SNR 3 is:")
    print(SNR3)

    recon_img4 = image[point_img3:,point2:point3,:]
    real_img4 = image[point_img3:,point3:,:]

    SNR4 = compute_SNR(real_img4, recon_img4)
    print("Calculated SNR 4 is:")
    print(SNR4)


    temp = tf.reshape(image,[1,dim1,dim2,dim3])
    filename = 'batch%06d_%s.png' % (batch, suffix)
    summary_op = tf.summary.image(filename, temp)
    summary_run = td.sess.run(summary_op)
    sum_writer.add_summary(summary_run,1)






    #print(tf.trainable_variables())
    #var = [v for v in tf.trainable_variables() if v.name == "gene_layer/encoder_5_2/batchnorm/batchnorm/add_1:0"][0]
    latent = tf.get_default_graph().get_tensor_by_name('gene_layer/encoder_5_2/batchnorm/batchnorm/add_1:0')
    #print(latent)

    ### LATENT SPACE HERE
    latent_space = td.sess.run(latent,feed_dict = feed_dict)
    # print(latent_space.shape)
    # print(type(latent_space))
    # print(np.linalg.norm(latent_space))
    # print(latent_space)
    # print(np.max(latent_space))
    # print(np.min(latent_space))

    
    filename = 'latent_batch%06d_%s.out' % (batch, suffix)
    filename = os.path.join('latent_arrays',filename)
    np.save(filename,latent_space)

    ### Latent stuff (messes everything up)

"""
    logs_path = 'tensorboard'
    with open(os.path.join(logs_path, "s-metadata.tsv"), 'w') as metadata_file:
        for row in range(2):
            c = row
            metadata_file.write('{}\n'.format(c))

    metadata = 's-metadata.tsv'

    images = tf.Variable(latent_space, name= 'images')
    sess.run(tf.global_variables_initializer())

    config = projector.ProjectorConfig()
    config.model_checkpoint_path = logs_path
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    embedding.metadata_path = metadata
    projector.visualize_embeddings(sum_writer,config)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(logs_path, "model.ckpt"), 1)

    """

    





    ###


"""
    if gene_param is not None:
        #add feature 
        print('dimension for input, ref, output:',
              feature.shape, label.shape, gene_output.shape)
        gene_param['feature'] = feature.tolist()
        gene_param['label'] = label.tolist()
        gene_param['eta'] = [x.tolist() for x in eta]
        gene_param['nmse'] = [x.tolist() for x in nmse]
        gene_param['kappa'] = [x.tolist() for x in kappa]

        #gene_param['gene_output'] = gene_output.tolist()
        #gene_param['gene_output_save'] = gene_output_save.tolist()
        #add input arguments
        #print(FLAGS.__dict__['__flags'])
        gene_param['FLAGS'] = FLAGS.__dict__['__flags']

        # save json
        filename = 'batch%06d_%s.json' % (batch, suffix)
        filename = os.path.join(FLAGS.train_dir, filename)
        with open(filename, 'w') as outfile:
            json.dump(gene_param, outfile)
        print("    Saved %s" % (filename,))
"""

def _save_checkpoint(train_data, batch):
    td = train_data

    oldname = 'checkpoint_old.txt'
    newname = 'checkpoint_new.txt'

    oldname = os.path.join(FLAGS.checkpoint_dir, oldname)
    newname = os.path.join(FLAGS.checkpoint_dir, newname)

    # Delete oldest checkpoint
    try:
        tf.gfile.Remove(oldname)
        tf.gfile.Remove(oldname + '.meta')
    except:
        pass

    # Rename old checkpoint
    try:
        tf.gfile.Rename(newname, oldname)
        tf.gfile.Rename(newname + '.meta', oldname + '.meta')
    except:
        pass

    # Generate new checkpoint
    saver = tf.train.Saver(sharded=True)
    saver.save(td.sess, newname)

    print("Checkpoint saved")

def train_model(sess,train_data, num_sample_train=1984, num_sample_test=116):
    
    td = train_data
    summary_op = td.summary_op

    # update merge_all_summaries() to tf.summary.merge_all
    # summaries = tf.summary.merge_all()
    # td.sess.run(tf.initialize_all_variables()) # will deprecated 2017-03-02
    # DONE: change to tf.global_variables_initializer()
    
    #commented May 10, 2018
    # td.sess.run(tf.global_variables_initializer())

    #TODO: load data

    lrval = FLAGS.learning_rate_start
    start_time = time.time()
    done = False
    batch = FLAGS.starting_batch

    # batch info    
    batch_size = FLAGS.batch_size

    num_batch_train = num_sample_train / batch_size
    num_batch_test = num_sample_test / batch_size            

    # learning rate
    assert FLAGS.learning_rate_half_life % 10 == 0

    # Cache test features and labels (they are small)    
    # update: get all test features
    list_test_features = []
    list_test_labels = []
    for batch_test in range(int(num_batch_test)):
        test_feature, test_label = td.sess.run([td.test_features, td.test_labels])
        list_test_features.append(test_feature)
        list_test_labels.append(test_label)
    print('prepare {0} test feature batches'.format(num_batch_test))
    # print([type(x) for x in list_test_features])
    # print([type(x) for x in list_test_labels])
    accumuated_err_loss=[]
 
    #tensorboard summary writer
    sum_writer=tf.summary.FileWriter(FLAGS.tensorboard_dir, td.sess.graph)

    while not done:
        batch += 1
        gene_ls_loss = gene_dc_loss = gene_loss = gene_mse_loss = disc_real_loss = disc_fake_loss = -1.234



        #first train based on MSE and then GAN
        if batch < 1e4:
           feed_dict = {td.learning_rate : lrval, td.gene_mse_factor : 1.0,td.train_phase: True}
        else:
           feed_dict = {td.learning_rate : lrval, td.gene_mse_factor : 1.0}  #1/np.sqrt(batch+100-1e3) + 0.9}
        #feed_dict = {td.learning_rate : lrval}
        
        # for training 
        # don't export var and layers for train to reduce size
        # move to later
        # ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.disc_real_loss, td.disc_fake_loss, 
        #        td.train_features, td.train_labels, td.gene_output]#, td.gene_var_list, td.gene_layers]
        # _, _, gene_loss, disc_real_loss, disc_fake_loss, train_feature, train_label, train_output = td.sess.run(ops, feed_dict=feed_dict)
        ops = [td.gene_minimize, td.disc_minimize, summary_op, td.gene_loss, td.gene_mse_loss, td.gene_ls_loss, td.gene_dc_loss, td.disc_real_loss, td.disc_fake_loss, td.list_gene_losses]                   
        _, _, fet_sum, gene_loss, gene_mse_loss, gene_ls_loss, gene_dc_loss, disc_real_loss, disc_fake_loss, list_gene_losses = td.sess.run(ops, feed_dict=feed_dict)
        
        sum_writer.add_summary(fet_sum,batch)


        # get all losses
        list_gene_losses = [float(x) for x in list_gene_losses]
    
        # verbose training progress
        if batch % 100 == 0:
            # Show we are alive
            elapsed = int(time.time() - start_time)/60
            err_log = 'Progress[{0:3f}%], ETA[{1:4f}m], Batch [{2:4f}], G_MSE_Loss[{3}], G_DC_Loss[{4:5f}], G_LS_Loss[{5:3.3f}], D_Real_Loss[{6:3.3f}], D_Fake_Loss[{7:3.3f}]'.format(
                    int(100*elapsed/FLAGS.train_time), FLAGS.train_time - elapsed, batch, 
                    gene_mse_loss, gene_dc_loss, gene_ls_loss, disc_real_loss, disc_fake_loss)
            print(err_log)
            # update err loss
            err_loss = [int(batch), float(gene_loss), float(gene_dc_loss), 
                        float(gene_ls_loss), float(disc_real_loss), float(disc_fake_loss)]
            accumuated_err_loss.append(err_loss)
            # Finished?            
            current_progress = elapsed / FLAGS.train_time
            if (current_progress >= 1.0) or (batch > FLAGS.train_time*200):
                done = True
            
            # Update learning rate
            if batch % FLAGS.learning_rate_half_life == 0:
                lrval *= .5
        # export test batches

        if batch % FLAGS.summary_period == 0:
            # loop different test batch
            for index_batch_test in range(int(num_batch_test)):

                # get test feature
                test_feature = list_test_features[index_batch_test]
                test_label = list_test_labels[index_batch_test]
            
                # Show progress with test features
                feed_dict = {td.gene_minput: test_feature, td.label_minput: test_label,td.train_phase: False}
                # not export var
                # ops = [td.gene_moutput, td.gene_mlayers, td.gene_var_list, td.disc_var_list, td.disc_layers]
                # gene_output, gene_layers, gene_var_list, disc_var_list, disc_layers= td.sess.run(ops, feed_dict=feed_dict)       
                
                ops = [td.gene_moutput, td.gene_moutput_list, td.gene_mlayers_list, td.gene_mask_list, td.gene_mask_list_0, td.disc_layers, td.eta, td.nmse, td.kappa]
                
                # get timing
                forward_passing_time = time.time()
                gene_output, gene_output_list, gene_layers_list, gene_mask_list, gene_mask_list_0, disc_layers, eta, nmse, kappa= td.sess.run(ops, feed_dict=feed_dict)       
                inference_time = time.time() - forward_passing_time

                # print('gene_var_list',[x.shape for x in gene_var_list])
                #print('gene_layers',[x.shape for x in gene_layers])
                # print('disc_var_list',[x.shape for x in disc_var_list])
                #print('disc_layers',[x.shape for x in disc_layers])

                # try to reduce mem
                #gene_output = None
                #gene_layers = None
                #disc_layers = None
                #accumuated_err_loss = []

                # save record
                gene_param = {'train_log':err_log,
                              'train_loss':accumuated_err_loss,
                              'gene_loss':list_gene_losses,
                              'inference_time':inference_time,
                              'gene_output_list':[x.tolist() for x in gene_output_list], 
                              'gene_mask_list':[x.tolist() for x in gene_mask_list],
                              'gene_mask_list_0':[x.tolist() for x in gene_mask_list_0]} #,
                              #'gene_mlayers_list':[x.tolist() for x in gene_layers_list]} 
                              #'disc_layers':[x.tolist() for x in disc_layers]}                
                # gene layers are too large
                #if index_batch_test>0:
                    #gene_param['gene_layers']=[]
                print("at summarizing stage")
                feed_dict = {td.gene_minput: test_feature, td.label_minput: test_label,td.train_phase: True}
                _summarize_progress(sess,feed_dict,sum_writer,td, test_feature, test_label, gene_output, gene_output_list, eta, nmse, kappa, batch,  
                                    'test{0}'.format(index_batch_test),                                     
                                    max_samples = FLAGS.batch_size,
                                    gene_param = gene_param)
                # try to reduce mem
                gene_output = None
                gene_layers = None
                disc_layers = None
                accumuated_err_loss = []

        # export train batches
        if OUTPUT_TRAIN_SAMPLES and (batch % FLAGS.summary_train_period == 0):
            # get train data
            ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.gene_ls_loss, td.gene_dc_loss, td.disc_real_loss, td.disc_fake_loss, 
                   td.train_features, td.train_labels, td.gene_output]#, td.gene_var_list, td.gene_layers]
            _, _, gene_loss, gene_dc_loss, gene_ls_loss, disc_real_loss, disc_fake_loss, train_feature, train_label, train_output, mask = td.sess.run(ops, feed_dict=feed_dict)
            print('train sample size:',train_feature.shape, train_label.shape, train_output.shape)
            _summarize_progress(sess,feed_dict,sum_writer,td, train_feature, train_label, train_output, batch%num_batch_train, 'train')

        
        

        # export check points
        if batch % FLAGS.checkpoint_period == 0:
            # Save checkpoint
            _save_checkpoint(td, batch)

    _save_checkpoint(td, batch)
    print('Finished training!')

