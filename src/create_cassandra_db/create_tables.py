"""This Python file enables the creation of
tables in Apache Cassandra"""

from cassandra.cluster import Session

from .constants import ENCODING


def create_table(
        session: Session,
        table_name: str
) -> None:
    """
    Create a table to persist speech data given an input session and name.

    Args:
        session: Session
            An Apache Cassandra DB session.
        table_name: str
            The name of the table to be created.
    """
    query = (f"CREATE TABLE IF NOT EXISTS {table_name} "
             f"(subject_id text, jitter_percent float, "
             f"jitter_abs float, rap float, ppq float, "
             f"apq_3 float, apq_5 float, apq_11 float, status int, "
             f"PRIMARY KEY (subject_id))")
    session.execute(query)


def write_data_to_table(
        session: Session,
        path_to_csv_file: str,
        table_name: str) -> None:
    """
    Write speech data into a table given an input session and csv file path.

    Args:
        session: Session
            An Apache Cassandra DB session.
        path_to_csv_file: str
            The full path to a csv file.
        table_name: str
            The name of the table of interest.
    """
    with open(path_to_csv_file, 'r', encoding=ENCODING) as input_file:
        i = 1
        for line_number, line in enumerate(input_file):
            if line_number == 0:
                continue  # Skip the first line, as it has the header with the column names
            row = line.replace('\n', "").split(',')

            query = f"INSERT INTO {table_name} (subject_id, jitter_percent, jitter_abs, rap, ppq, \
                                       apq_3, apq_5, apq_11, status)"
            query = query + " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            session.execute(query, (
                str(row[0]), float(row[1]), float(row[2]), float(row[3]),
                float(row[4]), float(row[5]), float(row[6]), float(row[7]),
                int(row[8])))
            i = i + 1
