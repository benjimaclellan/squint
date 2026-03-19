
#%%
from plum import dispatch
from jaxtyping import ArrayLike
from beartype import beartype


class AbstractBackend:
    pass

class DynamiqsBackend(AbstractBackend):
    pass

class TensorNetworkBackend(AbstractBackend):
    pass