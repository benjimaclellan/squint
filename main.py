# %%
import pathlib

import typer

app = typer.Typer()


@app.command()
def noise(
    path: pathlib.Path, filename: str, n: int, state: str, channel: str, loc: str
):
    from squint.cli.noise import noise

    noise(path=path, filename=filename,  n=n, state=state, channel=channel,loc=loc)


@app.command()
def telescope(path: str, n: int):
    from squint.cli.telescope import telescope

    telescope(path, n)


if __name__ == "__main__":
    app()
