import h5py
import mxnet as mx
import argparse
import os
import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from model import I3D_MX_Simple, I3D_MX_TF


######################################################################
## Load parameters from HDF5 to Dict
######################################################################
def load_conv3d(arg_params, aux_params, modality, dump_dir, name_tf, name_pth=""):
    h5f = h5py.File(dump_dir + modality + "/" + name_pth + name_tf + "/" + name_tf.split('/')[-1] + '.h5', 'r')

    root_pth = modality + "/" + name_pth + name_tf

    # weight need (out_channel, in_channel, L, H, W)
    weight = mx.nd.array(h5f['weights'][()].transpose(4, 3, 0, 1, 2))
    out_planes = weight.shape[0]
    gamma = mx.nd.ones(out_planes)
    beta = mx.nd.array(h5f['beta'][()].reshape(-1))
    mean = mx.nd.array(h5f['mean'][()].reshape(-1))
    var = mx.nd.array(h5f['var'][()].reshape(-1))

    assert arg_params[root_pth + '/conv_3d/_weight'].shape == weight.shape
    assert arg_params[root_pth + '/batch_norm/_gamma'].shape == gamma.shape
    assert arg_params[root_pth + '/batch_norm/_beta'].shape == beta.shape
    assert aux_params[root_pth + '/batch_norm/_moving_mean'].shape == mean.shape
    assert aux_params[root_pth + '/batch_norm/_moving_var'].shape == var.shape

    arg_params[root_pth + '/conv_3d/_weight'] = weight
    # print(name_pth, state_dict[name_pth+'.conv.weight'].size())
    # TODO why ones?
    arg_params[root_pth + '/batch_norm/_gamma'] = gamma
    arg_params[root_pth + '/batch_norm/_beta'] = beta
    aux_params[root_pth + '/batch_norm/_moving_mean'] = mean
    aux_params[root_pth + '/batch_norm/_moving_var'] = var
    h5f.close()


def load_conv3_Mixed_5b(arg_params, aux_params, modality, dump_dir, name_tf, name_pth=""):
    h5f = h5py.File(dump_dir + modality + "/" + name_pth + name_tf + "/" + name_tf.split('/')[-1] + '.h5', 'r')

    root_pth = modality + "/" + name_pth + "Conv3d_0b_3x3"

    # weight need (out_channel, in_channel, L, H, W)
    weight = mx.nd.array(h5f['weights'][()].transpose(4, 3, 0, 1, 2))
    out_planes = weight.shape[0]
    gamma = mx.nd.ones(out_planes)
    beta = mx.nd.array(h5f['beta'][()].reshape(-1))
    mean = mx.nd.array(h5f['mean'][()].reshape(-1))
    var = mx.nd.array(h5f['var'][()].reshape(-1))

    assert arg_params[root_pth + '/conv_3d/_weight'].shape == weight.shape
    assert arg_params[root_pth + '/batch_norm/_gamma'].shape == gamma.shape
    assert arg_params[root_pth + '/batch_norm/_beta'].shape == beta.shape
    assert aux_params[root_pth + '/batch_norm/_moving_mean'].shape == mean.shape
    assert aux_params[root_pth + '/batch_norm/_moving_var'].shape == var.shape

    arg_params[root_pth + '/conv_3d/_weight'] = weight
    # print(name_pth, state_dict[name_pth+'.conv.weight'].size())
    # TODO why ones?
    arg_params[root_pth + '/batch_norm/_gamma'] = gamma
    arg_params[root_pth + '/batch_norm/_beta'] = beta
    aux_params[root_pth + '/batch_norm/_moving_mean'] = mean
    aux_params[root_pth + '/batch_norm/_moving_var'] = var
    h5f.close()


def load_Mixed(arg_params, aux_params, modality, dump_dir='./data/dump', name='Mixed_3b'):
    load_conv3d(arg_params, aux_params, modality, dump_dir, name_pth=name + "/Branch_0/", name_tf='Conv3d_0a_1x1')
    load_conv3d(arg_params, aux_params, modality, dump_dir, name_pth=name + "/Branch_1/", name_tf='Conv3d_0a_1x1')
    load_conv3d(arg_params, aux_params, modality, dump_dir, name_pth=name + "/Branch_1/", name_tf='Conv3d_0b_3x3')
    load_conv3d(arg_params, aux_params, modality, dump_dir, name_pth=name + "/Branch_2/", name_tf='Conv3d_0a_1x1')
    load_conv3d(arg_params, aux_params, modality, dump_dir, name_pth=name + "/Branch_2/", name_tf='Conv3d_0b_3x3')
    load_conv3d(arg_params, aux_params, modality, dump_dir, name_pth=name + "/Branch_3/", name_tf='Conv3d_0b_1x1')


def load_Mixed_5b(arg_params, aux_params, modality, dump_dir='./data/dump', name='Mixed_5b'):
    load_conv3d(arg_params, aux_params, modality, dump_dir, name_pth=name + "/Branch_0/", name_tf='Conv3d_0a_1x1')
    load_conv3d(arg_params, aux_params, modality, dump_dir, name_pth=name + "/Branch_1/", name_tf='Conv3d_0a_1x1')
    load_conv3d(arg_params, aux_params, modality, dump_dir, name_pth=name + "/Branch_1/", name_tf='Conv3d_0b_3x3')
    load_conv3d(arg_params, aux_params, modality, dump_dir, name_pth=name + "/Branch_2/", name_tf='Conv3d_0a_1x1')
    # load_conv3d(arg_params, aux_params, modality, dump_dir, name_pth = name + "/Branch_2/" , name_tf='Conv3d_0b_3x3')

    # since original repo's typo
    load_conv3_Mixed_5b(arg_params, aux_params, modality, dump_dir, name_pth=name + "/Branch_2/",
                        name_tf='Conv3d_0a_3x3')
    load_conv3d(arg_params, aux_params, modality, dump_dir, name_pth=name + "/Branch_3/", name_tf='Conv3d_0b_1x1')


def load_Logits(arg_params, aux_params, modality, dump_dir, name_tf, name_pth=''):
    h5f = h5py.File(dump_dir + modality + "/" + name_tf + "/" + name_tf.split('/')[-1] + '.h5', 'r')

    root_pth = modality + "/" + name_pth + name_tf
    weight = mx.nd.array(h5f['weights'][()].transpose(4, 3, 0, 1, 2))
    bias = mx.nd.array(h5f['bias'][()].reshape(-1))

    assert arg_params[root_pth + '/conv_3d/_weight'].shape == weight.shape
    assert arg_params[root_pth + '/conv_3d/_bias'].shape == bias.shape

    # weight need (out_channel, in_channel, L, H, W)
    arg_params[root_pth + '/conv_3d/_weight'] = weight
    arg_params[root_pth + '/conv_3d/_bias'] = bias

    h5f.close()

def save(mod, prefix):
    mod._symbol.save('%s-symbol.json' % prefix)
    mod.save_params('%s.params' % prefix)

def load(prefix):
    symbol = mx.sym.load('%s-symbol.json' % prefix)
    save_dict = mx.nd.load('%s.params' % prefix)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        #print (name)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return (symbol, arg_params, aux_params)

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    # Model Parmeters
    parser.add_argument('--dump_dir', type=str, default='../data/dump/',
                        help='source directory')
    parser.add_argument('--target_dir', type=str, default=None,
                        help='source directory')
    parser.add_argument('--modality', type=str, default='flow_imagenet',
                        help='rgb_scratch , rgb_imagenet, flow_scratch, flow_imagenet')
    parser.add_argument('--model', type=str, default='I3D_MX_TF',
                        help='I3D_MX_TF or I3D_MX_Simple')
    parser.add_argument('--pooling_convention', type=str, default='full',
                        help='full or valid')
    args = parser.parse_args()
    
    if args.target_dir is None:

        if os.path.exists("../data/mxnet_checkpoints/") == False:
            os.mkdir("../data/mxnet_checkpoints/")
        if os.path.exists("../data/mxnet_checkpoints/" +args.modality + "/") == False:
            os.mkdir("../data/mxnet_checkpoints/" +args.modality + "/")

        target_dir = "../data/mxnet_checkpoints/" +args.modality + "/" + args.modality

    else:

        if os.path.exists(args.target_dir) == False:
            os.mkdir(args.target_dir)

        target_dir = args.target_dir


    modality = args.modality
    dump_dir = args.dump_dir

    # create a trainable module on GPU 0
    i3d = eval(args.model + "." + args.model)(modality=modality, pooling_convention=args.pooling_convention)

    predictions = i3d.get_I3D()
    mod = mx.mod.Module(symbol=predictions, context=mx.cpu())

    if 'rgb' in modality:
        mod.bind(data_shapes=[('data', (1, 3, 79, 224, 224))], force_rebind=True)
    elif 'flow' in modality:
        print("in flow")
        mod.bind(data_shapes=[('data', (1, 2, 79, 224, 224))], force_rebind=True)
    mod.init_params(mx.initializer.Uniform(scale=1.0))

    mod.save_checkpoint('temp', 0)
    sym, arg_params, aux_params = mx.model.load_checkpoint('temp', 0)

    load_conv3d(arg_params, aux_params, modality, dump_dir, name_tf='Conv3d_1a_7x7')
    load_conv3d(arg_params, aux_params, modality, dump_dir, name_tf='Conv3d_2b_1x1')
    load_conv3d(arg_params, aux_params, modality, dump_dir, name_tf='Conv3d_2c_3x3')
    load_Mixed(arg_params, aux_params, modality, dump_dir, name='Mixed_3b')
    load_Mixed(arg_params, aux_params, modality, dump_dir, name='Mixed_3c')
    load_Mixed(arg_params, aux_params, modality, dump_dir, name='Mixed_4b')
    load_Mixed(arg_params, aux_params, modality, dump_dir, name='Mixed_4c')
    load_Mixed(arg_params, aux_params, modality, dump_dir, name='Mixed_4d')
    load_Mixed(arg_params, aux_params, modality, dump_dir, name='Mixed_4e')
    load_Mixed(arg_params, aux_params, modality, dump_dir, name='Mixed_4f')
    load_Mixed_5b(arg_params, aux_params, modality, dump_dir, name='Mixed_5b')
    load_Mixed(arg_params, aux_params, modality, dump_dir, name='Mixed_5c')
    load_Logits(arg_params, aux_params, modality, dump_dir, name_tf='Logits/Conv3d_0c_1x1')

    mod.set_params(arg_params, aux_params)
    mod.save_checkpoint(target_dir,0)
