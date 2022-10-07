
import tensorflow as tf
import utils
from google.protobuf import text_format
from models.research.object_detection.utils import config_util
from object_detection.protos import pipeline_pb2


# Accessing configs
config = config_util.get_configs_from_pipeline_file(utils.ConfigPath)

# editing the pipeline config file of our new OD model
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(utils.ConfigPath, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = 26
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = utils.TrainedModelsPath+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= utils.AnnotationPath + '/Signs_label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [utils.AnnotationPath + '/test_signs.tfrecord']
pipeline_config.eval_input_reader[0].label_map_path = utils.AnnotationPath + '/Signs_label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [utils.AnnotationPath + '/train_signs.tfrecord']

#Outputs the pipeline pbtxt so we can see if it has correctly updated
pipeline_config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(utils.ConfigPath, "wb") as f:                                                                                                                                                                                                                     
    f.write(pipeline_config_text)

#Train
###In Command Line Interface in the Deep Learning directory run:    (Remember to change the model)
####### python models/research/object_detection/model_main_tf2.py --model_dir= RealTimeObjectDetection/workspace/models/asl_alphabet_mobnet --pipeline_config_path=RealTimeObjectDetection/workspace/models/asl_alphabet_mobnet/pipeline.config --num_train_steps=10000