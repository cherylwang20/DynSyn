class DynSynWeightAmpFunc:
    def relu_like(num_timesteps, k, a):
        dynsyn_weight_amp = max(0, k * (num_timesteps - a))
        dynsyn_weight_amp = min(dynsyn_weight_amp, 0.1)
        return dynsyn_weight_amp


def get_dynsyn_weight_amp(num_timesteps, dynsyn_k, dynsyn_a, dynsyn_weight_amp):
    dynsyn_weight_amp = (
        dynsyn_weight_amp
        if dynsyn_weight_amp is not None
        else DynSynWeightAmpFunc.relu_like(num_timesteps, dynsyn_k, dynsyn_a)
    )
    return dynsyn_weight_amp
