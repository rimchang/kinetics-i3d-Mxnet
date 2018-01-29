import mxnet as mx
from .I3D_utils import get_Conv3dTF, get_MaxPool3dTF

class I3D_MX_TF:
    def __init__(self, modality, pooling_convention='full', num_classes=400, squeeze=True):

        self.modality = modality
        self.squeeze = squeeze
        self.num_classes = num_classes
        self.eps = 1e-3
        self.pooling_convention = pooling_convention

    def BasicConv3d(self, data, num_filter, kernel, stride=(1, 1, 1), padding="SAME", dilate=(1, 1, 1),
                    use_global_stats=True,
                    name='Conv3d_1a_7x7'):

        conv = get_Conv3dTF(data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                            dilate=dilate, padding=padding, no_bias=True, name='%s/conv_3d/' % name)
        bn = mx.sym.BatchNorm(data=conv, eps=self.eps, momentum=0.001, fix_gamma=True,
                              use_global_stats=use_global_stats, name='%s/batch_norm/' % name)
        act = mx.sym.Activation(data=bn, act_type='relu', name='%s/relu/' % name)

        return act

    def get_Mixed(self, data, dilate=(1, 1, 1), filter_list=None, name='Mixed_3c'):

        if filter_list is None:
            raise Exception("need filter list")

        # branch 0
        branch_0 = self.BasicConv3d(data=data, num_filter=filter_list[0], kernel=(1, 1, 1), stride=(1, 1, 1),
                                    name=('%s/%s/Branch_0/Conv3d_0a_1x1' % (self.modality, name)))

        # branch 1
        branch_1 = self.BasicConv3d(data=data, num_filter=filter_list[1], kernel=(1, 1, 1), stride=(1, 1, 1),
                                    name=('%s/%s/Branch_1/Conv3d_0a_1x1' % (self.modality, name)))
        branch_1 = self.BasicConv3d(data=branch_1, num_filter=filter_list[2], kernel=(3, 3, 3), stride=(1, 1, 1),
                                    dilate=dilate, name=('%s/%s/Branch_1/Conv3d_0b_3x3' % (self.modality, name)))

        # branch 2
        branch_2 = self.BasicConv3d(data=data, num_filter=filter_list[3], kernel=(1, 1, 1), stride=(1, 1, 1),
                                    name=('%s/%s/Branch_2/Conv3d_0a_1x1' % (self.modality, name)))
        branch_2 = self.BasicConv3d(data=branch_2, num_filter=filter_list[4], kernel=(3, 3, 3), stride=(1, 1, 1),
                                    dilate=dilate, name=('%s/%s/Branch_2/Conv3d_0b_3x3' % (self.modality, name)))

        # branch 3

        branch_3 = get_MaxPool3dTF(data=data, kernel=(3, 3, 3), stride=(1, 1, 1), pooling_convention=self.pooling_convention,
                                   name=('%s/%s/Branch_2/MaxPool3d_0a_3x3' % (self.modality, name)))
        branch_3 = self.BasicConv3d(data=branch_3, num_filter=filter_list[5], kernel=(1, 1, 1), stride=(1, 1, 1),
                                    dilate=dilate, name=('%s/%s/Branch_3/Conv3d_0b_1x1' % (self.modality, name)))

        # concat
        concat = mx.sym.Concat(*[branch_0, branch_1, branch_2, branch_3], dim=1,
                               name=('%s/%s/Concat' % (self.modality, name)))
        return concat


    def get_I3D(self):

        data = mx.sym.Variable(name="data")
        Conv3d_1a_7x7 = self.BasicConv3d(data=data, num_filter=64, kernel=(7, 7, 7), stride=(2, 2, 2),
                                         name=('%s/Conv3d_1a_7x7' % (self.modality)))

        MaxPool3d_2a_3x3 = get_MaxPool3dTF(data=Conv3d_1a_7x7, kernel=(1, 3, 3), stride=(1, 2, 2), pooling_convention=self.pooling_convention,
                                           name=('%s/MaxPool3d_2a_3x3/' % (self.modality)))

        Conv3d_2b_1x1 = self.BasicConv3d(data=MaxPool3d_2a_3x3, num_filter=64, kernel=(1, 1, 1), stride=(1, 1, 1),
                                         name=('%s/Conv3d_2b_1x1' % (self.modality)))
        Conv3d_2c_3x3 = self.BasicConv3d(data=Conv3d_2b_1x1, num_filter=192, kernel=(3, 3, 3), stride=(1, 1, 1),
                                         name=('%s/Conv3d_2c_3x3' % (self.modality)))

        MaxPool3d_3a_3x3 = get_MaxPool3dTF(data=Conv3d_2c_3x3, kernel=(1, 3, 3), stride=(1, 2, 2), pooling_convention=self.pooling_convention,
                                           name=('%s/MaxPool3d_3a_3x3/' % (self.modality)))

        mixed_3b = self.get_Mixed(MaxPool3d_3a_3x3, filter_list=[64, 96, 128, 16, 32, 32],
                                  name='Mixed_3b')
        mixed_3c = self.get_Mixed(mixed_3b, filter_list=[128, 128, 192, 32, 96, 64], name='Mixed_3c')

        MaxPool3d_4a_3x3 = get_MaxPool3dTF(data=mixed_3c, kernel=(3, 3, 3), stride=(2, 2, 2), pooling_convention=self.pooling_convention,
                                           name=('%s/MaxPool3d_4a_3x3/' % (self.modality)))

        mixed_4b = self.get_Mixed(MaxPool3d_4a_3x3, filter_list=[192, 96, 208, 16, 48, 64],
                                  name='Mixed_4b')
        mixed_4c = self.get_Mixed(mixed_4b, filter_list=[160, 112, 224, 24, 64, 64], name='Mixed_4c')
        mixed_4d = self.get_Mixed(mixed_4c, filter_list=[128, 128, 256, 24, 64, 64], name='Mixed_4d')
        mixed_4e = self.get_Mixed(mixed_4d, filter_list=[112, 144, 288, 32, 64, 64], name='Mixed_4e')
        mixed_4f = self.get_Mixed(mixed_4e, filter_list=[256, 160, 320, 32, 128, 128],
                                  name='Mixed_4f')

        MaxPool3d_5a_2x2 = get_MaxPool3dTF(data=mixed_4f, kernel=(2, 2, 2), stride=(2, 2, 2), pooling_convention=self.pooling_convention,
                                              name=('%s/MaxPool3d_5a_2x2/' % (self.modality)))

        mixed_5b = self.get_Mixed(MaxPool3d_5a_2x2, filter_list=[256, 160, 320, 32, 128, 128],
                                  name='Mixed_5b')
        mixed_5c = self.get_Mixed(mixed_5b, filter_list=[384, 192, 384, 48, 128, 128],
                                  name='Mixed_5c')

        avg_pooling = mx.sym.Pooling(data=mixed_5c, kernel=(2, 7, 7), stride=(1, 1, 1), global_pool=False,
                                     pool_type='avg',
                                     name=('%s/Logits/average_pooling/' % (self.modality)))
        drop_out = mx.symbol.Dropout(data=avg_pooling,
                                     name=('%s/Logits/drop_out/' % (self.modality)))
        logits = mx.sym.Convolution(data=drop_out, num_filter=self.num_classes, kernel=(1, 1, 1), stride=(1, 1, 1),
                                    name=('%s/Logits/Conv3d_0c_1x1/conv_3d/' % (self.modality)))

        if self.squeeze:
            logits = mx.sym.split(logits, axis=3, num_outputs=1, squeeze_axis=True)
            logits = mx.sym.split(logits, axis=3, num_outputs=1, squeeze_axis=True)


        return logits
