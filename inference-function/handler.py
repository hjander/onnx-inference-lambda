import json
import base64
import time
import onnxruntime as rt
import numpy as np

from PIL import Image
from io import BytesIO

# adpated from https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/resnet50_modelzoo_onnxruntime_inference.ipynb

def inference(event, context):

    print(event)

    result = run(event)

    response = {
        "statusCode": 200,
        "body": json.dumps(result)
    }

    return response



def load_labels(path):

    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)


def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

    #add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()


def read_image_from_body(event):

    fileBody = json.loads(event['body'])['file']
    #return Image.open(BytesIO(base64.b64decode(fileBody)))
    img = Image.open("/opt/7214.jpg")
    print(img.size)
    return img.resize((224,224))

def run(input_data_json):
    # try:
        labels = load_labels('/opt/imagenet_class_index.json')
        image = read_image_from_body(input_data_json)
        # image = Image.open('images/plane.jpg')

        image_data = np.array(image).transpose(2, 0, 1)
        input_data = preprocess(image_data)

        sess_options = rt.SessionOptions()
        sess_options.enable_profiling = True

        start = time.time()

        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = 8
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_profiling = True

        session = rt.InferenceSession('/opt/model.onnx', sess_options=sess_options)
        input_name = session.get_inputs()[0].name
        raw_result = session.run([], {input_name: input_data})

        end = time.time()
        res = postprocess(raw_result)

        inference_time = np.round((end - start) * 1000, 2)
        idx = np.argmax(res)

        sort_idx = np.flip(np.squeeze(np.argsort(res)))

        result = dict()
        #result['top_prediction'] = labels[idx]
        #result['top5_prediction'] = labels[sort_idx[:5]]
        result['image_size_bytes'] = image.size
        result['inference_time_ms'] = str(inference_time)

        print(result)

        return json.dumps(result)

    # except Exception as e:
    #     result = str(e)
    #     return {"error": result}
