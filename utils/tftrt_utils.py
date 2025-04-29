# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 22:45:52 2020

@author: deeplens
"""
import keras
from keras.models import load_model
from keras import backend as K

import tensorflow as tf
import uff
#import tensorflow.contrib.tensorrt as trt
from tensorflow.contrib import tensorrt as tftrt

from tensorflow.python.platform import gfile
import numpy as np
import pdb
import copy

class FrozenGraph(object):
  def __init__(self, model, shape):
    shape = (None, shape[0], shape[1], shape[2])
    x_name = 'image_tensor_x'
    with K.get_session() as sess:
        x_tensor = tf.placeholder(tf.float32, shape, x_name)
        K.set_learning_phase(0)
        y_tensor = model(x_tensor)
        y_name = y_tensor.name[:-2]
        graph = sess.graph.as_graph_def()
        graph0 = tf.graph_util.convert_variables_to_constants(sess, graph, [y_name])
        graph1 = tf.graph_util.remove_training_nodes(graph0)

    self.x_name = [x_name]
    self.y_name = [y_name]
    self.frozen = graph1

class TfEngine(object):
  def __init__(self, graph):
    g = tf.Graph()
    with g.as_default():
      x_op, y_op = tf.import_graph_def(
          graph_def=graph.frozen, return_elements=graph.x_name + graph.y_name)
      self.x_tensor = x_op.outputs[0]
      self.y_tensor = y_op.outputs[0]

    config = tf.ConfigProto(gpu_options=
      tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
      allow_growth=True))

    self.sess = tf.Session(graph=g, config=config)

  def infer(self, x):
    y = self.sess.run(self.y_tensor,
      feed_dict={self.x_tensor: x})
    return y

class TftrtEngine(TfEngine):
  def __init__(self, graph, batch_size, precision):
    tftrt_graph = tftrt.create_inference_graph(
      graph.frozen,
      outputs=graph.y_name,
      max_batch_size=batch_size,
      max_workspace_size_bytes=1 << 30,
      precision_mode=precision,
      minimum_segment_size=2)

    opt_graph = copy.deepcopy(graph)
    opt_graph.frozen = tftrt_graph
    super(TftrtEngine, self).__init__(opt_graph)
    self.batch_size = batch_size

  def infer_clas(self, x):
    num_tests = x.shape[0]
    y = np.empty((num_tests, self.y_tensor.shape[1]), np.float32)
    batch_size = self.batch_size

    for i in range(0, num_tests, batch_size):
      x_part = x[i : i + batch_size]
      y_part = self.sess.run(self.y_tensor,
        feed_dict={self.x_tensor: x_part})
      y[i : i + batch_size] = y_part
    return y
    
  def infera_seg(self, x):
    num_tests = x.shape[0]
    y = np.empty((num_tests, self.y_tensor.shape[1],self.y_tensor.shape[2],self.y_tensor.shape[3]), np.float32)
    batch_size = self.batch_size

    for i in range(0, num_tests, batch_size):
      x_part = x[i : i + batch_size]
      y_part = self.sess.run(self.y_tensor,
        feed_dict={self.x_tensor: x_part})
      y[i : i + batch_size] = y_part
    return y
    
def verify(predict, ans):
    passed = 0
    num_test = ans.shape[0]
    
    for i in range(0, num_test):
        a = ans[i]
        p = np.argmax(predict[i])
        if (p == a) : passed = passed + 1
    
    if (float(passed) / num_test > 0.99) : print('PASSED')
    else : print('FAILURE', passed)
    
    p = np.argmax(predict[0])
    print('first inference result:', p, '\n\n')
    
def convert_fg2uff(frozen_graph_path,output_names):    
    output_filename = frozen_graph_path[:frozen_graph_path.rfind('.')]  + '.uff'
    trt_graph = uff.from_tensorflow_frozen_model(frozen_graph_path, [output_names])   
    print('Writing  UFF to disk...')
    with open(output_filename, 'wb') as f:
        f.write(trt_graph)
    print('UFF saving is done')   
    # check how many ops that is converted to TensorRT engine
    #trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
    #print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
    #all_nodes = len([1 for n in trt_graph.node])
    #print("numb. of all_nodes in TensorRT graph:", all_nodes)

def convert_fg_TRT(frozen_graph,output_names):
    
    trt_graph = tftrt.create_inference_graph(
    input_graph_def=frozen_graph,# frozen model
    outputs=output_names,
    max_batch_size=2,# specify your max batch size
    max_workspace_size_bytes=2*(10**9),# specify the max workspace
    precision_mode="FP16") # precision, can be "FP32" (32 floating point precision) or "FP16"

    #write the TensorRT model to be used later for inference
    with gfile.FastGFile("./models/TensorRT_model.pb", 'wb') as f:
        f.write(trt_graph.SerializeToString())
    print("TensorRT model is successfully stored!")
    
    # check how many ops of the original frozen model
    all_nodes = len([1 for n in frozen_graph.node])
    print("numb. of all_nodes in frozen graph:", all_nodes)
    
    # check how many ops that is converted to TensorRT engine
    trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
    print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
    all_nodes = len([1 for n in trt_graph.node])
    print("numb. of all_nodes in TensorRT graph:", all_nodes)

    

def save_frozen_graph_and_UFF(model, filename):
    # First freeze the graph and remove training nodes.
    output_names = model.output.op.name
    sess = tf.keras.backend.get_session()
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_names])
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
    
      
    # check how many ops of the original frozen model
    all_nodes = len([1 for n in frozen_graph.node])
    print("numb. of all_nodes in frozen graph:", all_nodes)
    
    # Save the model
    with open(filename, "wb") as ofile:
        ofile.write(frozen_graph.SerializeToString())   
    
    print('Frozen graph saving is done')
    
    frozen_graph_path = filename
    #pdb.set_trace()
    print('Input_name :'+model.input.op.name )
    print('Output_name :'+model.output.op.name)
    convert_fg2uff(frozen_graph_path,output_names)
    #convert_fg_TRT(frozen_graph,output_names)

#def main():
#    x_train, y_train, x_test, y_test = process_dataset()
#    model = create_model()
#    model.summary()
#    # Train the model on the data
#    model.fit(x_train, y_train, epochs = 5, verbose = 1)
#    # Evaluate the model on test data
#    model.evaluate(x_test, y_test)
#    model.save('models/lenet5_5.pb')
#    save(model, filename="models/lenet5_5.pb")
#
#if __name__ == '__main__':
#    main()
