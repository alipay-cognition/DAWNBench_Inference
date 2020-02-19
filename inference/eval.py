import os
import sys
import gflags
from types import MethodType
import tensorflow as tf
from inception_preprocessing import preprocess_image as inception_preprocess_image
from vgg_preprocessing import preprocess_image as vgg_preprocess_image
import hgai;hgai.tf_init()
from hgai.frontend.tensorflow import converter
from PIL import Image
import numpy as np
import time

Flags = gflags.FLAGS 
gflags.DEFINE_string('pb_model_file', './model/resnet_v1_101.npu.pb', 'model file of pb') 
gflags.DEFINE_integer('num_steps', 1000, "Number of steps to evaluate, default:1000") 
gflags.DEFINE_integer('num_warmup', 10, "Number of steps to warmup, default:10") 
gflags.DEFINE_integer('intra_op_parallelism_threads', 1, "inter_op_parallelism_threads, default:1") 
gflags.DEFINE_integer('inter_op_parallelism_threads', 10, "inter_op_parallelism_threads, default:10") 
gflags.DEFINE_string('validation_path', './imagenet_val/tf_record/', 'validation tf recode') 

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

def get_files(data_dir, filename_pattern):
    if data_dir == None:
        return []
    files = tf.gfile.Glob(os.path.join(data_dir, filename_pattern))
    if files == []:
        raise ValueError('Can not find any files in {} with pattern "{}"'.format(
            data_dir, filename_pattern))
    return files

def get_tfrecords_count(files):
    num_records = 0
    for fn in files:
        for record in tf.python_io.tf_record_iterator(fn):
            num_records += 1
    return num_records

def read_and_decode(record):
    features = tf.parse_single_example(record,features={
        'image/encoded': tf.FixedLenFeature([], tf.string, ''),
        'image/class/label': tf.FixedLenFeature([1], tf.int64, -1),
        'image/class/text': tf.FixedLenFeature([], tf.string, ''),
        })
    image_buffer = features['image/encoded']
    try:
        image = tf.image.decode_jpeg(image_buffer, 
                                     channels=IMG_CHANNELS, 
                                     dct_method='INTEGER_FAST')
    except:
        image = tf.image.decode_png(imgdata, channels=3)
    label = tf.cast(features['image/class/label'],tf.int32)
    image = vgg_preprocess_image(image,
                                 IMG_HEIGHT, IMG_WIDTH,
                                 is_training=False)
    image = tf.expand_dims(image, 0)
    return image,label


class NpuModel(object):
    def __init__(self):
        pass

    def load_model(self, model_file):
        self.graph = tf.Graph()
        self.graph_def = tf.GraphDef()
        with open(model_file, "rb") as f:
            self.graph_def.ParseFromString(f.read())
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def)

        self.model_name = "resnet_v1_101"
        self.config = {
            'input_nodes': ['input'],
            'input_shapes': [[1, 224, 224, 3]],
            'output_nodes': ['logits', 'classes'],
        }
        self.model_info = {
            'image_hw': 224,
            'preprocess': 'inception',
            'num_classes': 1001,
            'slim': True,
        }
        tf_config = tf.ConfigProto()
        tf_config.intra_op_parallelism_threads = Flags.intra_op_parallelism_threads;
        tf_config.inter_op_parallelism_threads = Flags.inter_op_parallelism_threads;
        self.session = tf.Session(graph = self.graph,
                                  config = tf_config)

        self.input_tensor = self.graph.get_tensor_by_name("import/input:0")
        self.outputs_tensor = [
            self.graph.get_tensor_by_name("import/logits:0")
        ]
        
    def get_graph(self):
        return self.graph_def
        
    def get_config(self):
        return self.config
        
    def run(self, image_np):
        input_params_dict = {
            self.input_tensor: image_np
        }
        result = self.session.run(self.outputs_tensor,
                                  feed_dict=input_params_dict)
        return result
        
def main(argv):
    Flags(argv)
    
    validation_files = get_files(Flags.validation_path, "validation*")
    all_count = get_tfrecords_count(validation_files)
    all_count = min(Flags.num_steps, all_count)
    print("num_steps:%d" %(all_count))
    print("num_warmup:%d" %(Flags.num_warmup))
    
    dataset = tf.data.TFRecordDataset(validation_files)
    dataset = dataset.map(read_and_decode)
    
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    
    images = []
    labels = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(all_count):
            image,label = sess.run(next_batch)
            images.append(image)
            labels.append(label)
    npu_model = NpuModel()
    npu_model.load_model(Flags.pb_model_file)

    top_k = 5
    labels_result = []
    all_rt = []
    sum_rt = 0
    for image in images:
        t1= time.time()
        result = npu_model.run(image)
        t2 = time.time()
        all_rt.append(t2 - t1)
        sort_index = np.argsort(-result[0])
        labels_result.append((sort_index[0][:top_k] + 1))
    top5_success = 0
    top1_success = 0
    for idx in range(len(labels)):
        label = labels[idx]
        label_result = labels_result[idx]
        if label in label_result:
            top5_success +=1
        if label == label_result[0]:
            top1_success +=1
    print("top1:%f" %(top1_success/len(labels)))
    print("top5:%f" %(top5_success/len(labels)))
    print("Latency mean:%fms" %(np.mean(np.array(all_rt[Flags.num_warmup:])) * 1000) )
    
if __name__ == "__main__":
    main(sys.argv)
    
