#%%
from pydantic import BaseModel, Field
import jax.numpy as jnp
import numpy as np
from squint.cli.telescope import TelescopeArgs as TelescopeArgs
from squint.cli.noise import NoiseArgs as NoiseArgs
from typing import Union, Literal, Optional
from jaxtyping import PyTree, ArrayLike
import pathlib
import h5py
import io
import equinox as eqx
#%%
class Result(BaseModel):
    args: Union[TelescopeArgs, NoiseArgs]
    circuit: Optional[PyTree] = Field(None, exclude=True)
    datasets: Optional[dict[str, ArrayLike]] = Field(default_factory=lambda: {}, exclude=True)

    class Config:
        arbitrary_types_allowed = True
        
        
    def save(self, filepath: pathlib.Path):
        assert self.circuit is not None, "Circuit must be provided"

    # def save(filepath: pathlib.Path, args: BaseModel, model: eqx.Module, datasets: dict[str, ArrayLike]):
        with h5py.File(filepath, "a") as f:
            f.attrs['schema'] = self.model_dump_json()
            buf = io.BytesIO()
            eqx.tree_serialise_leaves(buf, self.circuit)
            buf.seek(0)
            
            if "circuit" in f.keys():
                del f['circuit']
            f.create_dataset("circuit", data=np.void(buf.getvalue()))
            
            for key, value in self.datasets.items():
                if key in f.keys():
                    del f[key]
                f.create_dataset(key, data=value)


    @classmethod
    def load(cls, filepath: pathlib.Path):
        data = {}
        with h5py.File(filepath, "r") as f:
            args_json = f.attrs['schema']
            result = cls.model_validate_json(args_json)

            model = result.args.make()
            for key in f.keys():
                if key == 'circuit':
                    buf = io.BytesIO(f["circuit"][()].tobytes())
                    deserialized = eqx.tree_deserialise_leaves(buf, model)
                    result.circuit = deserialized
                    continue
                
                result.datasets[key] = jnp.array(f[key][()])
            
            return result

# %%
if __name__ == "__main__":
#%%
    path = pathlib.Path(f"data/2025-03-22-test")
    filename = 'noise' 

    args = NoiseArgs(path=path, filename=filename, n=4, state='ghz', loc='state', channel='bitflip')

    result = Result(args=args)
    result.circuit = args.make()
    result.datasets = {'a': jnp.ones([10, 10])}
    
    result.model_dump_json()
    
    result.save("test.h5")
    
    #%%
    result_load = Result.load("test.h5")
# %%
