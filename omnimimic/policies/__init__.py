from .convnet_policy import GCConvPolicy
from .act_policy import ACTPolicy
from .gnm_policy import GNMPolicy, GNMNumpyWrapper

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(**policy_config)
    elif policy_class == 'ConvNet':
        policy = GCConvPolicy(**policy_config)
    elif policy_class == 'GNM':
        policy = GNMPolicy(**policy_config)
    return policy

