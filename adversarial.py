class AdversarialImage(object):

    def __init__(self,inp,out,eps=0.01):
        '''
        inp : input tensor  (image)
        out : output tensor (y_pred)
        eps : scalar
        '''
        self.inp = inp.outputs[0]
        self.out = out.outputs[0]
        self.define_aimage_tensor(float(eps))
        
    def mse_tf(self,y_pred,y_test, verbose=True):
        '''
        y_pred : tensor 
        y_test : tensor having the same shape as y_pred
        '''
        ## element wise square
        minus  = tf.constant(-1.0)
        m_y_test = tf.scalar_mul(minus,y_test)
        square = tf.square(tf.add(y_pred ,m_y_test))## preserve the same shape as y_pred.shape
        ## mean across the final dimensions
        ms = tf.reduce_mean(square)
        return(ms)

    def define_aimage_tensor(self,eps):
        '''
        Define a graph to output adversarial image

        Xnew = X + eps * sign(dX)

        X : np.array of image of shape (None,height, width,n_channel)
        y : np.array containing the true landmark coordinates (None, 30)
        '''
        ## get list of target
        yshape = [None] + [int(i) for i in self.out.get_shape()[1:]]
        eps_tf = tf.constant(eps,name="epsilon")
        
        y_true_tf = tf.placeholder(tf.float32, yshape)
        y_pred_tf = self.out 

        loss = self.mse_tf(y_pred_tf,y_true_tf)

        ## tensor that calculate the gradient of loss with respect to image i.e., dX
        grad_tf          = tf.gradients(loss,[self.inp])
        grad_sign_tf     = tf.sign(grad_tf)
        grad_sign_eps_tf = tf.scalar_mul(eps_tf,
                                         grad_sign_tf)
        new_image_tf = tf.add(grad_sign_eps_tf,self.inp)
        
        self.y_true  = y_true_tf
        self.eps     = eps_tf
        self.aimage  = new_image_tf
        self.added_noise = grad_sign_eps_tf

    
    def predict(self,X):

        with tf.Session() as sess:
            y_pred = sess.run(self.out,
                              feed_dict={self.inp:X})
        return(y_pred)

        
    def get_aimage(self,X,y,added_noise=False):
        tensor2eval = [self.aimage]
        if added_noise:
            tensor2eval.append(self.added_noise)
            
        with tf.Session() as sess:
            result = sess.run(tensor2eval,
                              feed_dict={self.inp:X,
                                         self.y_true:y
                                         })
        for i in range(len(result)):
            result[i] = result[i].reshape(*X.shape)
        return(result)