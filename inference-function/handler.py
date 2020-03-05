import io
import json
import base64
import time

import onnxruntime as rt
import numpy as np

from PIL import Image


# adapted from:
# https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/resnet50_modelzoo_onnxruntime_inference.ipynb
from requests_toolbelt.multipart import decoder


def inference(event, context):

    image = read_image_from_body(event)

    inference = OnnxModelInference('/opt/model.onnx', '/opt/imagenet_class_index.json')
    result = inference.infer(image)

    response = {
        "statusCode": 200,
        "body": json.dumps(result)
    }

    return response


class OnnxModelInference(object):

    def __init__(self, model_path, model_labels_path):

        sess_options = rt.SessionOptions()
        sess_options.enable_profiling = True

        sess_options.intra_op_num_threads = 8
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_profiling = True

        start = time.time()
        self.inference_session = rt.InferenceSession(model_path, sess_options=sess_options)
        self.__load_model_ms = time.time() - start

        self.labels = self.load_labels(model_labels_path)

    def infer(self, image):

        try:
            resized_img = image.resize((224, 224))
            image_data = np.array(resized_img).transpose(2, 0, 1)
            input_data = self.preprocess(image_data)

            start = time.time()

            input_name = self.inference_session.get_inputs()[0].name
            raw_result = self.inference_session.run([], {input_name: input_data})

            end = time.time()
            res = self.postprocess(raw_result)

            inference_time = np.round((end - start) * 1000, 2)
            idx = np.argmax(res)

            sort_idx = np.flip(np.squeeze(np.argsort(res)))

            result = dict()
            result['top_prediction'] = self.labels[idx]
            result['top5_prediction'] = self.labels[sort_idx[:5]].tolist()
            result['load_time_ms'] = self.__load_model_ms
            result['inference_time_ms'] = str(inference_time)
            result['device'] = rt.get_device()
            #result['session_options'] = self.inference_session.get_session_options()

            return json.dumps(result)

        except Exception as e:
            result = str(e)
            return {"error": result}

    def preprocess(self, input_data):

        # convert the input data into the float32 input
        img_data = input_data.astype('float32')

        # normalize
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])

        norm_img_data = np.zeros(img_data.shape).astype('float32')
        for i in range(img_data.shape[0]):
            norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

        # add batch channel
        norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
        return norm_img_data

    def softmax(self, x):
        x = x.reshape(-1)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def postprocess(self, result):
        return self.softmax(np.array(result)).tolist()

    def load_labels(self,path):
            with open(path) as f:
                data = json.load(f)

                label_list = list()
                for entry in data:
                    label_list.append(list(data[entry])[1])


            return np.asarray(label_list)


def read_image_from_body(event):

    body = event["body"]

    body_dec = base64.b64decode(body)
    imageStream = io.BytesIO(body_dec)
    imageFile = Image.open(imageStream)

    print("Image decoded")
    return imageFile
