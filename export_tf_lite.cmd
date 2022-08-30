python %OBJ_DETECT%\export_tflite_graph_tf2.py --trained_checkpoint_dir out_export\checkpoint --output_directory out_tflite_intermediate --pipeline_config_path out_export\pipeline.config
mkdir out_tflite
mkdir out_tflite_intermediate
python convert_to_tflite.py --tf_lite_model=out_tflite/model.tflite --tf_dataset=dataset/train.tfrecord --intermed_dir=out_tflite_intermediate/saved_model
python add_tflite_metadata.py --tf_lite_model=out_tflite/model.tflite --tf_lite_model_out=out_tflite/model_with_metadata.tflite --labels_map=labels_map.pbtxt --labels_map_out=out_tflite/tflite_labels_map.txt
