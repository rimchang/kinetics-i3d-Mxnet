import mxnet as mx
import argparse
import numpy as np

from convert.convert_mxnet import load
from model.I3D_MX import InceptionI3d_MX

_IMAGE_SIZE = 224
_NUM_CLASSES = 400

_SAMPLE_VIDEO_FRAMES = 79
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/mxnet_checkpoints/rgb_scratch/rgb_scratch',
    'flow': 'data/mxnet_checkpoints/flow_scratch/flow_scratch',
    'rgb_imagenet': 'data/mxnet_checkpoints/rgb_imagenet/rgb_imagenet',
    'flow_imagenet': 'data/mxnet_checkpoints/flow_imagenet/flow_imagenet',
}

_LABEL_MAP_PATH = 'data/label_map.txt'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Model Parmeters
    parser.add_argument('--eval_type', type=str, default='rgb',
                        help='rgb, flow, or joint')
    parser.add_argument('--imagenet_pretrained', type=str2bool, default='True')
    args = parser.parse_args()

    eval_type = args.eval_type

    if eval_type not in ['rgb', 'flow', 'joint']:
        raise ValueError('Bad `eval_type`, must be one of rgb, flow, joint')


    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

    rgb_logits, flow_logits = (0,0)

    if eval_type in ['rgb', 'joint']:

        if args.imagenet_pretrained:
            sym, arg_params, aux_params = mx.model.load_checkpoint(_CHECKPOINT_PATHS['rgb_imagenet'], 0)

            print('RGB checkpoint restored')
        else:
            sym, arg_params, aux_params = mx.model.load_checkpoint(_CHECKPOINT_PATHS['rgb'], 0)

            print('RGB checkpoint restored')

        mod = mx.mod.Module(symbol=sym, context=mx.cpu())
        mod.bind(data_shapes=[('data', (1, 3, 79, 224, 224))], force_rebind=True)

        mod.set_params(arg_params, aux_params)
        rgb_sample = mx.nd.array(np.load(_SAMPLE_PATHS['rgb']).transpose(0, 4, 1, 2 ,3))
        data_iter = mx.io.NDArrayIter(rgb_sample)
        print('RGB data loaded, shape=', str(rgb_sample.shape))

        rgb_logits = mod.predict(data_iter)
        #rgb_logits = mx.nd.mean(rgb_logits, axis=2)

    if eval_type in ['flow', 'joint']:

        if args.imagenet_pretrained:
            sym, arg_params, aux_params = mx.model.load_checkpoint(_CHECKPOINT_PATHS['flow_imagenet'], 0)

            print('flow checkpoint restored')
        else:
            sym, arg_params, aux_params = mx.model.load_checkpoint(_CHECKPOINT_PATHS['flow'], 0)

            print('flow checkpoint restored')

        mod = mx.mod.Module(symbol=sym, context=mx.cpu())
        mod.bind(data_shapes=[('data', (1, 2, 79, 224, 224))], force_rebind=True)

        mod.set_params(arg_params, aux_params)
        flow_sample = mx.nd.array(np.load(_SAMPLE_PATHS['flow']).transpose(0, 4, 1, 2 ,3))
        data_iter = mx.io.NDArrayIter(flow_sample)
        print('FLOW data loaded, shape=', str(flow_sample.shape))

        flow_logits = mod.predict(data_iter)
        #flow_logits = mx.nd.mean(flow_logits, axis=2)

    out_logits = rgb_logits + flow_logits
    out_logits = mx.nd.mean(out_logits, axis=2)
    out_predictions = mx.ndarray.softmax(out_logits, axis=1)

    sorted_indices = mx.nd.argsort(out_predictions, axis=1, is_ascend=False).asnumpy().astype('int').reshape(-1)

    out_logits = out_logits.asnumpy().reshape(-1)
    out_predictions = out_predictions.asnumpy().reshape(-1)


    print('Norm of logits: %f' % np.linalg.norm(out_logits))
    print('\nTop classes and probabilities')

    for index in sorted_indices[:20]:
        print(out_predictions[index], out_logits[index], kinetics_classes[index])
