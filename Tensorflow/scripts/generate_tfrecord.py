"""Sample TensorFlow XML-to-TFRecord converter"""

import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse
import warnings


# Suppress TensorFlow logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info and warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

# Argument parser setup
parser = argparse.ArgumentParser(description="Sample TensorFlow XML-to-TFRecord converter")
parser.add_argument("-x", "--xml_dir", help="Path to input .xml files.", type=str, required=True)
parser.add_argument("-l", "--labels_path", help="Path to the labels (.pbtxt) file.", type=str, required=True)
parser.add_argument("-o", "--output_path", help="Path of output TFRecord (.record) file.", type=str, required=True)
parser.add_argument("-i", "--image_dir", help="Path to input image files (defaults to XML_DIR).", type=str, default=None)
parser.add_argument("-c", "--csv_path", help="Path of output .csv file. If none, no file is written.", type=str, default=None)

args = parser.parse_args()

# Set image directory to XML directory if not provided
if args.image_dir is None:
    args.image_dir = args.xml_dir

# Confirm existence of label map file
label_map_path = args.labels_path
if not os.path.exists(label_map_path):
    raise FileNotFoundError(f"Label map file not found at {label_map_path}")

# Load label map and confirm successful load
try:
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    print("Loaded label map:", label_map_dict)
except Exception as e:
    print("Error loading label map:", e)
    exit(1)

def xml_to_csv(path):
    """Convert XML files to CSV."""
    xml_list = []
    for xml_file in glob.glob(os.path.join(path, '*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text))
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def class_text_to_int(row_label):
    """Map class text to integer using label map."""
    label_id = label_map_dict.get(row_label, None)
    if label_id is None:
        raise ValueError(f"Label '{row_label}' not found in label map. Check that all labels in XML files have corresponding entries in the label map file.")
    return label_id

def split(df, group):
    """Group data by filename."""
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    """Create a TensorFlow Example from grouped data."""
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    writer = tf.io.TFRecordWriter(args.output_path)
    path = os.path.join(args.image_dir)
    examples = xml_to_csv(args.xml_dir)
    grouped = split(examples, 'filename')
    
    # Write each example to TFRecord
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    
    writer.close()
    print('Successfully created the TFRecord file:', args.output_path)
    
    # Optionally write to CSV
    if args.csv_path is not None:
        examples.to_csv(args.csv_path, index=None)
        print('Successfully created the CSV file:', args.csv_path)

if __name__ == '__main__':
    main([])
