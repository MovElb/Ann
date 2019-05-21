import argparse

import torch
from flask import Flask, request, jsonify
from CloudModelWrapper import CloudModelWrapper
from utils import str2bool

app = Flask(__name__)
cmw = None


@app.route('/get_answer', methods=['POST'])
def get_model_answer():
    preprocessed_data = request.get_json()['data']
    model_answer = cmw.generate_model_answers(preprocessed_data)
    return jsonify(model_answer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--port', type=int, default=999)
    parser.add_argument("--use_cuda", type=str2bool, nargs='?',
                        const=True, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument("--model_path", default='./models/best_model.pt', help='path to model checkpoint')
    args = parser.parse_args()

    cmw = CloudModelWrapper(args)

    app.run('0.0.0.0', port=args.port)
