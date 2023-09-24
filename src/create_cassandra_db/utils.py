"""Utility-based functions to aid the creation
of keyspace and tables in Apache Cassandra"""

import pandas as pd
from cassandra.cluster import Cluster, Session
from cassandra.policies import DCAwareRoundRobinPolicy

from constants import DEFAULT_LOCAL_IP, KEYSPACE_NAME, PROTOCOL_VERSION_NUM


def create_session(ip_address: str = DEFAULT_LOCAL_IP) -> Session:
    """Create an Apache Cassandra session given an IP address.

    Args:
        ip_address: str
            Your local IP address ('127.0.0.1' by default).

    Returns:
        Session
            An Apache Cassandra DB session.
    """
    cluster = Cluster(
        [ip_address],
        protocol_version=PROTOCOL_VERSION_NUM,
        load_balancing_policy=DCAwareRoundRobinPolicy()
    )
    session = cluster.connect()
    return session


def create_and_set_keyspace(
        session: Session,
        ks_name: str = KEYSPACE_NAME
) -> None:
    """
    Create and set a keyspace given an input session and name.

    Args:
        session: Session
            An Apache Cassandra DB session.
        ks_name: str
            The name of a keyspace ('parkinson' by default).
    """
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS parkinson
        WITH REPLICATION =
        {'class': 'SimpleStrategy', 'replication_factor': 1}"""
                    )

    session.set_keyspace(ks_name)


def get_all_data_from_table(
        session: Session,
        table_name: str
) -> pd.DataFrame:
    """
    Get all speech data from a table given an input session and table name.

    Args:
        session: Session
            An Apache Cassandra DB session.
        table_name: str
            The name of the table of interest.

    Returns:
        pd.DataFrame
            A df with all speech data from a table.
    """
    all_rows = session.execute(f'select * from {table_name};')
    df_from_table = pd.DataFrame(list(all_rows))
    return df_from_table
