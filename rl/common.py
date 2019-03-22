""" Code common for all rl package. """
import tensorflow as tf


def flat_layers_iterator(model):
  """ Iterator over all layers of a given model. """
  layers = list(reversed(model.layers))
  while layers:
    toplayer = layers.pop()
    if hasattr(toplayer, "layers"):
      layers.extend(reversed(toplayer.layers))
    else:
      yield toplayer


def clone_layer(layer):
  """ Clones a given layer. """
  return layer.__class__.from_config(layer.get_config())


def clone_model(model, name=None):
  """ Clones a sequential model. """
  if not isinstance(model, tf.keras.Sequential):
    raise ValueError("can only copy models of type Sequential, got model "
                     f"type {type(model)}")

  def _is_int(s):
    # pylint: disable=invalid-name
    if s and s[0] in ('-', '+'):
      return s[1:].isdigit()
    return s.isdigit()

  if name is None:
    *name_parts, ending = model.name.split('_')
    if _is_int(ending):
      ending = int(ending) + 1
      name_parts.append(ending)
      name = '_'.join(name_parts)
    else:
      name_parts.append(ending)
      name_parts.append('copy')
      name = '_'.join(name_parts)

  # Use model._layers to ensure that all layers are cloned. The model's layers
  # property will exclude the initial InputLayer (if it exists) in the model,
  # resulting in a different Sequential model structure.
  # pylint: disable=protected-access
  layers = [clone_layer(layer) for layer in model._layers]
  return tf.keras.Sequential(layers, name=name)
