from .convnet_policy import GaussianCNNPolicy, GCConvPolicyWrapper
from .act_policy import ACTPolicy
from .gnm_policy import GNMPolicy, GNMNumpyWrapper


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(**policy_config)
    elif policy_class == 'ConvNet':
        policy = GaussianCNNPolicy(**policy_config)
    elif policy_class == 'GNM':
        policy = GNMPolicy(**policy_config)
    return policy

def get_numpy_wrapper(policy_class):
    if policy_class == 'GNM':
        wrapper = GNMNumpyWrapper
    elif policy_class == 'ConvNet':
        wrapper = GCConvPolicyWrapper
    return wrapper

