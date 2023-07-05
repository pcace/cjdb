import click
import os
from cjdb.logger import logger

from cjdb import __version__
from cjdb.modules.importer import Importer
from cjdb.modules.exporter import Exporter
from cjdb.modules.utils import get_db_engine, get_db_psycopg_conn
from cjdb.resources import strings as s


@click.group()
@click.version_option(
    __version__,
    "-v",
    "--version",
    prog_name="cjdb",
    message="cjdb " + __version__,
    help=s.version_help,
)
@click.pass_context
def cjdb(ctx):
    logger.info("cjdb importer/exporter!")


@cjdb.command(name="import")
@click.argument("filepath", type=str, default="stdin")
@click.option("--host", "-H", type=str, default="localhost", help=s.host_help)
@click.option("--port", "-p", type=int, default=5432, help=s.port_help)
@click.option("--user", "-U", type=str, required=True, help=s.user_help)
@click.password_option(
    help=s.password_help,
    prompt="Password for database user",
    confirmation_prompt=False
)
@click.option("--database", "-d",
              type=str,
              required=True,
              help=s.database_help)
@click.option("--schema", "-s", type=str, default="cjdb", help=s.schema_help)
@click.option("--append", "-a", type=bool, default=False, help=s.append_help)
@click.option("--srid", "-I", "target_srid",
              type=int,
              default=None,
              help=s.srid_help)
@click.option(
    "--attr-index", "-x",
    "indexed_attributes",
    type=str,
    multiple=True,
    help=s.index_help
)
@click.option(
    "--partial-attr-index",
    "-px",
    "partial_indexed_attributes",
    type=str,
    multiple=True,
    help=s.partial_index_help,
)
@click.option(
    "--ignore-repeated-file",
    "-g",
    "ignore_repeated_file",
    type=bool,
    is_flag=True,
    default=False,
    help=s.ignore_file_help,
)
@click.option(
    "--overwrite",
    "overwrite",
    is_flag=True,
    default=False,
    help=s.overwrite,
)
def import_cj(
    filepath,
    host,
    port,
    user,
    password,
    database,
    schema,
    append,
    target_srid,
    indexed_attributes,
    partial_indexed_attributes,
    ignore_repeated_file,
    overwrite,
):
    """Import CityJSONL files to a PostgreSQL database."""
    engine = get_db_engine(user, password, host, port, database)
    with Importer(
        engine,
        filepath,
        append,
        schema,
        target_srid,
        indexed_attributes,
        partial_indexed_attributes,
        ignore_repeated_file,
        overwrite,
    ) as imp:
        imp.run_import()


@cjdb.command(name="export")
@click.argument("query", type=str)
@click.option("--host", "-H", type=str, default="localhost", help=s.host_help)
@click.option("--port", "-p", type=int, default=5432, help=s.port_help)
@click.option("--user", "-U", type=str, default="postgres", help=s.user_help)
@click.password_option(
    help=s.password_help,
    prompt="Password for database user",
    confirmation_prompt=False
)
@click.option("--database", "-d",
              type=str,
              required=True,
              help=s.database_help)
@click.option("--schema", "-s", type=str, default="cjdb", help=s.schema_help)
@click.option("--output", "-o",
              type=str,
              default="cj_export.jsonl",
              help=s.output_help)
def export_cj(query, host, port, user, password, database, schema, output):
    """Export a CityJSONL stream to a file."""
    # where to save the file
    base = os.path.basename(output)
    dirname = os.path.abspath(os.path.dirname(output))
    # parent directory must exist
    if not os.path.exists(dirname):
        raise click.ClickException(
            'Output path does not exist: "%s"' % (dirname)
        )
    output_abs = os.path.join(dirname, base)
    conn = get_db_psycopg_conn(user, password, host, port, database)
    with Exporter(
        conn,
        schema,
        query,
        output_abs
    ) as exp:
        exp.run_export()


if __name__ == "__main__":
    cjdb()
