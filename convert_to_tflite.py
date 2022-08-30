from absl import app
from absl import flags
import tensorflow as tf
import numpy as np
import cv2


MODEL_INPUT_WIDTH_HEIGHT = 320
QUANT_REPRESENTATIVE_SAMPLES = 100


FLAGS = flags.FLAGS
flags.DEFINE_string("tf_lite_model", None, "Path to TF Lite model file, ex: out_tflite/model.tflite")
flags.DEFINE_string("tf_dataset", None, "Path to Tensorflow dataset, ex: dataset/train.tfrecord")
flags.DEFINE_string("intermed_dir", None, "Path to intermediate Tensorflow model directory, ex: out_tflite_intermediate/saved_model")
flags.mark_flag_as_required("tf_lite_model")
flags.mark_flag_as_required("tf_dataset")
flags.mark_flag_as_required("intermed_dir")


def parse_example(example):
    result = {}
    # example.features.feature is the dictionary
    for key, feature in example.features.feature.items():
        # The values are the Feature objects which contain a `kind` which contains:
        # one of three fields: bytes_list, float_list, int64_list

        kind = feature.WhichOneof('kind')
        result[key] = np.array(getattr(feature, kind).value)

    return result


# read representative dataset
def get_images():
    result = []
    for item in tf.data.TFRecordDataset(FLAGS.tf_dataset).take(QUANT_REPRESENTATIVE_SAMPLES):
        example = tf.train.Example()
        example.ParseFromString(item.numpy())
        example_dict = parse_example(example)
        img_data = example_dict["image/encoded"]
        img_data = img_data.view(np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
        # resize to 320x320
        img = cv2.resize(img, (MODEL_INPUT_WIDTH_HEIGHT, MODEL_INPUT_WIDTH_HEIGHT), interpolation = cv2.INTER_AREA)
        result.append(img)
    return result


def representative_dataset():
    for image in get_images():
        image = image.astype(np.float32)
        image = (image - 127.5)/127.5
        image = np.expand_dims(image, axis=0)
        yield [image]


def main(argv):
    del argv  # Unused

    converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.intermed_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16] # this line enables float16 quantization
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()

    with open(FLAGS.tf_lite_model, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    app.run(main)
