import tensorflow as tf
from tensorflow import keras
from .aleatoric import MVEWrapper
from .bias import HistogramWrapper
from .epistemic import VAEWrapper, DropoutWrapper, EnsembleWrapper
from .wrapper import Wrapper


def wrap(model, bias=True, aleatoric=True, epistemic=True, *args, **kwargs):
    """Abstract away the Wrapper and most parameters to simplify the wrapping process for the user."""
    metric_wrappers = []
    vae = None
    if bias == True:
        hist = VAEWrapper(model, is_standalone=False, epistemic=False)
        metric_wrappers.append(hist)
        vae = hist
    elif bias == False:
        pass
    elif type(bias) == list:
        for i in bias:
            out = _check_bias_compatibility(i, model)
            metric_wrappers.append(i)
            if type(i) == VAEWrapper:
                vae = i
    else:
        out = _check_bias_compatibility(bias)
        metric_wrappers.append(out)
        if type(i) == VAEWrapper:
            vae = i

    if aleatoric == True:
        metric_wrappers.append(MVEWrapper(model, is_standalone=False))
    elif aleatoric == False:
        pass
    elif type(aleatoric) == list:
        out = [_check_aleatoric_compatibility(i) for i in aleatoric]
        metric_wrappers.extend(out)
    else:
        out = _check_aleatoric_compatibility(aleatoric)
        if type(out) == type:
            out = out(model, is_standalone=False)
        metric_wrappers.append(out)

    if epistemic == True:
        if bias == False:
            metric_wrappers.append(VAEWrapper(model, is_standalone=False, bias=False))
    elif epistemic == False:
        pass
    elif type(epistemic) == list:
        out = [_check_epistemic_compatibility(i, model) for i in epistemic]
        for i in out:
            if type(i) == VAEWrapper and vae is not None:
                vae.epistemic = True
            else:
                metric_wrappers.append(i)
    else:
        out = _check_epistemic_compatibility(epistemic)
        if type(out) == VAEWrapper and vae is not None:
                vae.epistemic = True
        else:
            metric_wrappers.append(out)

    return Wrapper(model, metrics=metric_wrappers)


def _check_bias_compatibility(bias, model):
    bias_named_wrappers = {
        "HistogramWrapper": HistogramWrapper,
        "VAEWrapper" : VAEWrapper
    }
    add_vae = False
    if type(bias) == str and bias in bias_named_wrappers.keys():
        return bias_named_wrappers[bias](model, is_standalone=False)
    elif type(bias) == type and bias in bias_named_wrappers.values():
        return bias
    elif type(bias) in bias_named_wrappers.values():
        return bias
    else:
        raise ValueError(
            f"Must pass in either a string (one of {bias_named_wrappers.keys()}) or wrapper types (one of {bias_named_wrappers.values()}) or an instance of a wrapper type. Received {bias}"
        )


def _check_aleatoric_compatibility(aleatoric, model):
    aleatoric_named_wrappers = {"MVEWrapper": MVEWrapper}
    if type(aleatoric) == str and aleatoric in aleatoric_named_wrappers.keys():
        return aleatoric_named_wrappers[aleatoric](model, is_standalone=False)
    elif type(aleatoric) == type and aleatoric in aleatoric_named_wrappers.values():
        return aleatoric
    elif type(aleatoric) in aleatoric_named_wrappers.values():
        return aleatoric
    else:
        raise ValueError(
            f"Must pass in either a string (one of {aleatoric_named_wrappers.keys()}) or wrapper types (one of {aleatoric_named_wrappers.values()}) or an instance of a wrapper type. Received {aleatoric}"
        )


def _check_epistemic_compatibility(epistemic, model):
    epistemic_named_wrappers = {
        "DropoutWrapper": DropoutWrapper,
        "EnsembleWrapper": EnsembleWrapper,
        "VAEWrapper": VAEWrapper,
    }
    if type(epistemic) == str and epistemic in epistemic_named_wrappers.keys():
        return epistemic_named_wrappers[epistemic](model, is_standalone=False)
    elif type(epistemic) == type and epistemic in epistemic_named_wrappers.values():
        return epistemic
    elif type(epistemic) in epistemic_named_wrappers.values():
        return epistemic
    else:
        raise ValueError(
            f"Must pass in either a string (one of {epistemic_named_wrappers.keys()}) or wrapper types (one of {epistemic_named_wrappers.values()}) or an instance of a wrapper type. Received {epistemic}"
        )
