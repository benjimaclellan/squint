# %%
import pathlib

import typer

from squint.cli.result import Result


app = typer.Typer()


@app.command()
def noise(args: str):
    from squint.cli.noise import noise, NoiseArgs
    args = NoiseArgs.model_validate_json(args)
    # noise(path=path, filename=filename,  n=n, state=state, channel=channel,loc=loc)
    # circuit, args, datasets = noise(args=args)
    # result = Result(args=args, circuit=circuit, datasets=datasets)
    result = Result(**noise(args=args))
    result.save()

@app.command()
def telescope(args: str):
    from squint.cli.telescope import telescope, TelescopeArgs
    args = TelescopeArgs.model_validate_json(args)
    result = Result(**telescope(args=args))
    result.save()

    telescope(args=args)


if __name__ == "__main__":
    app()
