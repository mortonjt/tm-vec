import click
from click._compat import get_text_stderr
from click.exceptions import UsageError
from click.utils import echo


def _show_usage_error(self, file=None):
    if file is None:
        file = get_text_stderr()
    color = None
    if self.ctx is not None:
        color = self.ctx.color
        echo(self.ctx.get_help() + '\n', file=file, color=color)
    echo('Error: %s' % self.format_message(), file=file, color=color)


UsageError.show = _show_usage_error


@click.group()
@click.version_option()
def main():
    "tmvec: A tool for quick identification of remote homologues via protein embeddings"


# @main.command()

if __name__ == '__main__':
    main()
