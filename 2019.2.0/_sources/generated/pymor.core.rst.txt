pymor.core package
******************

.. automodule:: pymor.core
    :show-inheritance:

Submodules
==========

cache module
------------

.. automodule:: pymor.core.cache
    :show-inheritance:

---------


.. autoclass:: pymor.core.cache.CacheRegion
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.cache.CacheableInterface
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.cache.DiskRegion
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.cache.MemoryRegion
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autofunction:: pymor.core.cache.cached

---------


.. autofunction:: pymor.core.cache.cleanup_non_persisten_regions

---------


.. autofunction:: pymor.core.cache.clear_caches

---------


.. autofunction:: pymor.core.cache.default_regions

---------


.. autofunction:: pymor.core.cache.disable_caching

---------


.. autofunction:: pymor.core.cache.enable_caching

config module
-------------

.. automodule:: pymor.core.config
    :show-inheritance:

---------


.. autoclass:: pymor.core.config.Config
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autofunction:: pymor.core.config.is_jupyter

---------


.. autofunction:: pymor.core.config.is_nbconvert

---------


.. autofunction:: pymor.core.config.is_windows_platform

defaults module
---------------

.. automodule:: pymor.core.defaults
    :show-inheritance:

---------


.. autoclass:: pymor.core.defaults.DefaultContainer
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autofunction:: pymor.core.defaults.defaults

---------


.. autofunction:: pymor.core.defaults.defaults_sid

---------


.. autofunction:: pymor.core.defaults.load_defaults_from_file

---------


.. autofunction:: pymor.core.defaults.print_defaults

---------


.. autofunction:: pymor.core.defaults.set_defaults

---------


.. autofunction:: pymor.core.defaults.write_defaults_to_file

exceptions module
-----------------

.. automodule:: pymor.core.exceptions
    :show-inheritance:

---------


.. autoclass:: pymor.core.exceptions.AccuracyError
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.exceptions.ConstError
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.exceptions.ExtensionError
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.exceptions.GmshMissing
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.exceptions.ImageCollectionError
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.exceptions.InversionError
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.exceptions.LinAlgError
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.exceptions.MeshioMissing
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.exceptions.NewtonError
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.exceptions.NoMatchingRuleError
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.exceptions.QtMissing
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.exceptions.RuleNotMatchingError
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.exceptions.SIDGenerationError
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

interfaces module
-----------------

.. automodule:: pymor.core.interfaces
    :show-inheritance:

---------


.. autoclass:: pymor.core.interfaces.BasicInterface
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.interfaces.ImmutableInterface
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.interfaces.ImmutableMeta
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.interfaces.UID
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.interfaces.UberMeta
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.interfaces.classinstancemethod
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autofunction:: pymor.core.interfaces.generate_sid

logger module
-------------

.. automodule:: pymor.core.logger
    :show-inheritance:

---------


.. autoclass:: pymor.core.logger.ColoredFormatter
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.logger.DummyLogger
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autoclass:: pymor.core.logger.LogIndenter
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autofunction:: pymor.core.logger.default_handler

---------


.. autofunction:: pymor.core.logger.getLogger

---------


.. autofunction:: pymor.core.logger.set_log_format

---------


.. autofunction:: pymor.core.logger.set_log_levels

pickle module
-------------

.. automodule:: pymor.core.pickle
    :show-inheritance:

---------


.. autoclass:: pymor.core.pickle.Module
    :show-inheritance:
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__

---------


.. autofunction:: pymor.core.pickle._global_names

---------


.. autofunction:: pymor.core.pickle.dump

---------


.. autofunction:: pymor.core.pickle.dumps

---------


.. autofunction:: pymor.core.pickle.dumps_function

---------


.. autofunction:: pymor.core.pickle.load

---------


.. autofunction:: pymor.core.pickle.loads

---------


.. autofunction:: pymor.core.pickle.loads_function

