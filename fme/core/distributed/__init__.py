from .cooperative_stop import cooperative_stop
from .distributed import Distributed
from .shutdown import add_post_shutdown_callback

# `add_post_shutdown_callback` is exported alongside the other two because an
# adopter that wants work to run after the teardown -- writing a restart
# checkpoint, say -- should not have to reach into
# `fme.core.distributed.shutdown` for it. Those three names are the whole
# adopter-facing surface of the cooperative stop.
__all__ = ["Distributed", "add_post_shutdown_callback", "cooperative_stop"]
