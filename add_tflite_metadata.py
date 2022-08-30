from absl import app
from absl import flags
from object_detection.utils import label_map_util
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils
from tflite_support import metadata


FLAGS = flags.FLAGS
flags.DEFINE_string("tf_lite_model", None, "Path to TF Lite model file, ex: out_tflite/model.tflite")
flags.DEFINE_string("tf_lite_model_out", None, "Path to resulting TF Lite model file with metadata, ex: out_tflite/model_with_metadata.tflite")
flags.DEFINE_string("labels_map", None, "Path to Labels map file, ex: labels_map.pbtxt")
flags.DEFINE_string("labels_map_out", None, "Path to resulting (temporary) Labels map file, ex: out_tflite/tflite_labels_map.txt")
flags.mark_flag_as_required("tf_lite_model")
flags.mark_flag_as_required("tf_lite_model_out")
flags.mark_flag_as_required("labels_map")
flags.mark_flag_as_required("labels_map_out")


def main(argv):
    del argv  # Unused

    # We need to convert the Object Detection API's labelmap into what the Task API needs:
    # a txt file with one class name on each line from index 0 to N.
    # The first '0' class indicates the background.
    # This code assumes COCO detection which has 1 classes, you can write a label
    # map file for your model if re-trained.
    category_index = label_map_util.create_category_index_from_labelmap(FLAGS.labels_map)
    with open(FLAGS.labels_map_out, 'w') as f:
        for class_id in range(1, 2):
            if class_id not in category_index:
                f.write('???\n')
                continue
            name = category_index[class_id]['name']
            f.write(name+'\n')

    # Then we'll add the label map and other necessary metadata (e.g. normalization config) to the TFLite model.
    # As the SSD MobileNet V2 FPNLite 640x640 model take input image with pixel value in the range of [-1..1] (code),
    # we need to set norm_mean = 127.5 and norm_std = 127.5. See this documentation for more details on the normalization parameters.
    writer = object_detector.MetadataWriter.create_for_inference(
        writer_utils.load_file(FLAGS.tf_lite_model), input_norm_mean=[127.5],
        input_norm_std=[127.5], label_file_paths=[FLAGS.labels_map_out])
    writer_utils.save_file(writer.populate(), FLAGS.tf_lite_model_out)

    # Optional: Print out the metadata added to the TFLite model.
    displayer = metadata.MetadataDisplayer.with_model_file(FLAGS.tf_lite_model_out)
    print("Metadata populated:")
    print(displayer.get_metadata_json())
    print("=============================")
    print("Associated file(s) populated:")
    print(displayer.get_packed_associated_file_list())


if __name__ == '__main__':
    app.run(main)
