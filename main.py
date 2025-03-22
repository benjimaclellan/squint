# %%
import pathlib

import typer

app = typer.Typer()


@app.command()
def noise(
    path: pathlib.Path, filename: str, state: str, channel: str, n: int, loc: str
):
    from squint.cli.noise import noise

    noise(path=path, filename=filename, state=state, channel=channel, n=n, loc=loc)


@app.command()
def telescope(path: str, n: int):
    from squint.cli.telescope import telescope

    telescope(path, n)


if __name__ == "__main__":
    app()
