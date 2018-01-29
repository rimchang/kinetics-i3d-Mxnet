import mxnet as mx

def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)

    # pytorch pad : (paddingLeft, paddingRight, paddingTop, paddingBottom, paddingFront, paddingBack)
    # convert mxnet pad : (paddingFront, paddingBack, paddingTop, paddingBottom, paddingLeft, paddingRight)

    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, (padding_init, padding_init, padding_init)

def get_Conv3dTF(data, num_filter, kernel, stride=(1, 1, 1), dilate=(1, 1, 1), padding='SAME', no_bias=True,
                 name=""):
    if padding == 'SAME':
        padding_shape = get_padding_shape(kernel, stride)
        simplify_pad, simple_pad = simplify_padding(padding_shape)

        if simplify_pad:
            return mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, dilate=dilate, stride=stride,
                                      pad=simple_pad, no_bias=no_bias, name=name)
        else:
            pad_width = (0, 0, 0, 0) + padding_shape
            padded_data = mx.sym.pad(data, mode="constant", constant_value=0, pad_width=pad_width,
                                     name=name + "/pad")

            return mx.sym.Convolution(data=padded_data, num_filter=num_filter, kernel=kernel, dilate=dilate,
                                      stride=stride, pad=(0, 0, 0), no_bias=no_bias, name=name)

    elif padding == 'VALID':
        return mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, dilate=dilate, stride=stride,
                                  pad=(0, 0, 0), no_bias=no_bias, name=name)

def get_MaxPool3dTF(data, kernel, stride=(1, 1, 1), padding='SAME', pooling_convention='valid', name=""):
    if padding == 'SAME':
        padding_shape = get_padding_shape(kernel, stride)
        simplify_pad, simple_pad = simplify_padding(padding_shape)

        if simplify_pad:
            return mx.sym.Pooling(data=data, kernel=kernel, stride=stride, pad=simple_pad,
                                  pool_type='max', pooling_convention=pooling_convention, name=name)
        else:
            pad_width = (0, 0, 0, 0) + padding_shape
            padded_data = mx.sym.pad(data, mode="constant", constant_value=0, pad_width=pad_width,
                                     name=name + "/pad")
            return mx.sym.Pooling(data=padded_data, kernel=kernel, stride=stride, pad=(0, 0, 0),
                                  pool_type='max', pooling_convention=pooling_convention, name=name)

    elif padding == 'VALID':
        return mx.sym.Pooling(data=data, kernel=kernel, stride=stride, pad=(0, 0, 0), pool_type='max', pooling_convention=pooling_convention,
                                  name=name)