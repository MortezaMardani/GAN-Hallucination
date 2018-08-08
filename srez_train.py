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

def _summarize_progress(td,gene_output,train_feature,train_label,batch,index_batch_test):
    print("summarizing")

    gene_output = gene_output[0]

    size = [train_label.shape[1], train_label.shape[2]]
    print(size)

    # complex input zpad into r and channel
    complex_zpad = tf.image.resize_nearest_neighbor(train_feature, size)
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




    label_complex = tf.complex(train_label[:,:,:,0], train_label[:,:,:,1])
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
    filename = 'magnitude_batch%06d%d.out' % (batch,index_batch_test)
    filename = os.path.join('mags',filename)
    np.save(filename,image)

    
    # 3rd channel for visualization
    mag_3rd = np.maximum(image[:,:,0],image[:,:,1])
    image = np.concatenate((image, mag_3rd[:,:,np.newaxis]),axis=2)

    # save to image file
    print('save to image,', image.shape)
    filename = 'batch%06d%d.png' % (batch,index_batch_test)
    filename = os.path.join(FLAGS.train_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))


def convert_to_image(gene_output,td):
    size = [FLAGS.sample_size,FLAGS.sample_size_y]
    gene_output = gene_output[0]
    print(gene_output.shape)
    real = gene_output[:,:,:,0]
    img = gene_output[:,:,:,1]
    gene_output_complex = tf.complex(real,img)
    mag_output = tf.maximum(tf.minimum(tf.abs(gene_output_complex), 1.0), 0.0)
    mag_output = tf.reshape(mag_output, [FLAGS.batch_size, size[0], size[1], 1])
    mag_output = tf.concat(axis=3, values=[mag_output, mag_output])

    #concatenate and add third channel for visualization
    image = mag_output
    image = image[0:FLAGS.batch_size,:,:,:]
    image = tf.concat(axis=0, values=[image[i,:,:,:] for i in range(int(FLAGS.batch_size))])
    image  = td.sess.run(image)
    mag_3rd = np.maximum(image[:,:,0],image[:,:,1])
    image = np.concatenate((image, mag_3rd[:,:,np.newaxis]),axis=2)
    return image



def generate_new_images(train_data):
    n_latent = 512
    td = train_data
    shape = [FLAGS.batch_size,n_latent]
    num_to_generate = 10
    randoms = [np.random.normal(0, 1, shape) for _ in range(num_to_generate)]
    count = 1
    for vals in randoms:
        feed_dict = {td.train_phase: False,td.z_val: vals}
        ops = [td.gene_output]
        gen_output = td.sess.run(ops,feed_dict)
        image = convert_to_image(gen_output,td)

        #save image
        print('save to image,', image.shape)
        filename = 'image%d.png' % (count)
        filename = os.path.join("gen_images", filename)
        scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
        print("    Saved %s" % (filename,))
        count += 1







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
    list_train_features = []
    list_train_labels = []
    for batch_train in range(int(num_batch_train)):
        train_feature, train_label = td.sess.run([td.train_features, td.train_labels])
        list_train_features.append(train_feature)
        list_train_labels.append(train_label)
    print('prepare {0} test feature batches'.format(num_batch_train))
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
           feed_dict = {td.learning_rate : lrval, td.gene_mse_factor : 1.0, td.train_phase: True}  #1/np.sqrt(batch+100-1e3) + 0.9}
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




        ### This stuff is not needed for the VAE so make summary_period flag very high

        if batch % FLAGS.summary_period == 0:
            # loop different test batch
            for index_batch_test in range(10):

                # get test feature
                train_feature = list_train_features[index_batch_test]
                train_label = list_train_labels[index_batch_test]
            
                # Show progress with test features
                feed_dict = {td.gene_minput: train_feature,td.train_phase: True}
                # not export var
                # ops = [td.gene_moutput, td.gene_mlayers, td.gene_var_list, td.disc_var_list, td.disc_layers]
                # gene_output, gene_layers, gene_var_list, disc_var_list, disc_layers= td.sess.run(ops, feed_dict=feed_dict)       
                
                ops = [td.gene_output]
                
                # get timing
                forward_passing_time = time.time()
                gene_output = td.sess.run(ops, feed_dict=feed_dict)       
                inference_time = time.time() - forward_passing_time

                print("at summarizing stage")
                _summarize_progress(td, gene_output, train_feature, train_label,batch,index_batch_test)
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

    generate_new_images(td)


