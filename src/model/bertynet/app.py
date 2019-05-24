import argparse

import torch
from flask import Flask, request, jsonify
from bertynet.CloudModelWrapper import CloudModelWrapper
from bertynet.utils import str2bool

import sys

app = Flask(__name__)
cmw = None


@app.route('/get_answer', methods=['POST'])
def get_model_answer():
    preprocessed_data = request.get_json()['data']
    model_answer = cmw.generate_model_answers(preprocessed_data)
    return jsonify(model_answer)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--port', type=int, default=999)
    parser.add_argument("--use_cuda", type=str2bool, nargs='?',
                        const=True, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument("--model_path", default='./models/best_model.pt', help='path to model checkpoint')

    parser.add_argument('--meta_file', default='meta.msgpack', help='name of meta-data file')
    parser.add_argument('--data_dir', default='squad2_preprocessed', help='path to preprocessed data directory')

    args = parser.parse_args()

    print('Cuda is available:' + torch.cuda.is_available(), file=sys.stderr, flush=True)

    global cmw
    cmw = CloudModelWrapper(args)

    app.run('0.0.0.0', port=args.port)


if __name__ == '__main__':
    main()
