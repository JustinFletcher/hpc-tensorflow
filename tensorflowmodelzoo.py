import functools
import tensorflow as tf

# Import all models.
import lenet
import alexnet


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    Decorator source:
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    Decorator source:
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    Learning TensorFlow, pp 212.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


# function to print the tensor shape.  useful for debugging
def print_tensor_shape(tensor, string):
    '''
    input: tensor and string to describe it
    '''

    if __debug__:
        print('DEBUG ' + string, tensor.get_shape())


class TensorFlowModelZoo(object):

    def get_model(self, model_name, model_params=[]):

        if model_name == 'lenet':

            tfmodel = lenet.LeNetTensorFlowModel()

            return(tfmodel)

        if model_name == 'alexnet':

            tfmodel = alexnet.AlexNetTensorFlowModel()

            return(tfmodel)

        if model_name == 'vgg-16':

            print(model_name + " is not yet implemented.")
            raise NotImplementedError

        if model_name == 'googlenet':

            print(model_name + " is not yet implemented.")
            raise NotImplementedError

        if model_name == 'resnet-152':

            print(model_name + " is not yet implemented.")
            raise NotImplementedError

        else:

            print(model_name + " is not a recognized model name.")
            raise NotImplementedError

