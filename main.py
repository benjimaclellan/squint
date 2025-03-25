# %%
import pathlib
import os

os.environ['JAX_PLATFORMS'] = 'cpu'

import typer

from squint.cli.result import Result

app = typer.Typer()

@app.command()
def noise(path: pathlib.Path, filename: str, args: str):
    from squint.cli.noise import NoiseArgs, noise

    args = NoiseArgs.model_validate_json(args)
    circuit, args, datasets = noise(args=args)
    result = Result(args=args, circuit=circuit, datasets=datasets)
    result.save(path=path, filename=filename)


@app.command()
def telescope(path: pathlib.Path, filename: str, args: str):
    from squint.cli.telescope import TelescopeArgs, telescope

    args = TelescopeArgs.model_validate_json(args)
    circuit, args, datasets = telescope(args=args)
    result = Result(args=args, circuit=circuit, datasets=datasets)
    result.save(path=path, filename=filename)


if __name__ == "__main__":
    app()
