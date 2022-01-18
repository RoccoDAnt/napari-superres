
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from ._reader import napari_get_reader
from ._writer import napari_get_writer, napari_write_image
from ._dock_widget import napari_experimental_provide_dock_widget, ExampleQWidget, example_magic_widget
from ._function import napari_experimental_provide_function

from napari_plugin_engine import napari_hook_implementation

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    pass
#    from .napari_superres import launch_superres
#
#    return launch_superres
