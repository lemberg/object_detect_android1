import os, sys, glob
import random
import xml.etree.ElementTree as ET
import tensorflow as tf
import dataset_util

TEST_OUT_FILE = "test.tfrecord"
TRAIN_OUT_FILE = "train.tfrecord"
TRAIN_COEF = 0.9


def class_text_to_int(row_label):
    if row_label == 'wheel':
        return 1
    else:
        return None


def get_file_image_format(file):
    if file.endswith(".jpg"):
        return b'jpg'

    if file.endswith(".png"):
        return b'png'

    raise Exception("Unknown image format")


def xml_files_to_tfrecord(xml_files, out_file):
    with tf.io.TFRecordWriter(out_file) as writer:

        for xml_file in xml_files:
            try:
                xml_file_norm = os.path.normpath(xml_file)
                xml_file_path, xml_file_name = os.path.split(xml_file_norm)

                tree = ET.parse(xml_file)
                root = tree.getroot()

                filename = root[1].text
                path = root[2].text
                width = int(root[4][0].text)
                height = int(root[4][1].text)

                img_path = os.path.join(xml_file_path, path)

                xmins = []
                xmaxs = []
                ymins = []
                ymaxs = []
                classes_text = []
                classes = []

                for member in root.findall('object'):
                    obj_class = member[0].text
                    xmin = int(member[4][0].text)
                    ymin = int(member[4][1].text)
                    xmax = int(member[4][2].text)
                    ymax = int(member[4][3].text)

                    xmins.append(xmin/width)
                    xmaxs.append(xmax/width)
                    ymins.append(ymin/height)
                    ymaxs.append(ymax/height)
                    classes_text.append(obj_class.encode('utf8'))
                    classes.append(class_text_to_int(obj_class))

                with tf.io.gfile.GFile(img_path, 'rb') as fid:
                    encoded_image = fid.read()

                tf_example = tf.train.Example(features=tf.train.Features(feature={
                    'image/height': dataset_util.int64_feature(height),
                    'image/width': dataset_util.int64_feature(width),
                    'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
                    'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
                    'image/encoded': dataset_util.bytes_feature(encoded_image),
                    'image/format': dataset_util.bytes_feature(get_file_image_format(filename)),
                    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                    'image/object/class/label': dataset_util.int64_list_feature(classes),
                }))

                writer.write(tf_example.SerializeToString())

            except Exception as error:
                print("Failed to process file {}, Error: {}".format(xml_file, error))


def scan_xml_files(dirs):
    result = []
    for directory in dirs:
        print("Scanning directory {}...".format(directory))
        files_in_dir = glob.glob(directory + "/*.xml")
        result.extend(files_in_dir)
    return result


def main(dirs):
    xml_files = scan_xml_files(dirs)
    random.shuffle(xml_files)
    train_len = int(len(xml_files) * TRAIN_COEF)
    train_files = xml_files[0:train_len]
    test_files = xml_files[train_len:]

    print("Processing {}...".format(TRAIN_OUT_FILE))
    xml_files_to_tfrecord(train_files, TRAIN_OUT_FILE)
    print("Done. Processed {} images".format(len(train_files)))

    print("Processing {}...".format(TEST_OUT_FILE))
    xml_files_to_tfrecord(test_files, TEST_OUT_FILE)
    print("Done. Processed {} images".format(len(test_files)))


if len(sys.argv) < 1:
    print("No directories specified")
    exit(-1)
main(sys.argv)
