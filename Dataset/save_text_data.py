import tensorflow as tf
import json


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def save_text_to_json(source, obj_name, text_dict):
    """Saves text data scraped from the internet or other sources
       to a json file

       :param source: string of the data source
       :param obj_name: string of the object name
       :param text_dict: dictionary whose keys are key attributes of
                         a human-made space object, and whose values
                         are text summaries scraped from the internet
                         or other sources
    """
    json_file_name = source.lower() + '.txt'
    with open(json_file_name, 'a') as jf:
        data = dict()
        data[obj_name] = text_dict
        json.dump(data, jf)


def save_text_to_tfrecord(source, obj_name, text_dict):
    """Saves text data scaped from the internet or other sources
       to a tfrecord

       :param source: string of the data source
       :param obj_name: string of the object name
       :param text_dict: dictionary whose keys are key attributes of
                         a human-made space object, and whose values
                         are text summaries scraped from the internet
                         or other sources
    """
    # Build feature dict
    feature
