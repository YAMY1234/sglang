from enum import IntEnum, unique


@unique
class P2PTag(IntEnum):
    """
    Tags reserved for point-to-point communication protocols.

    Communications introduced outside existing scheduler loops need explicit
    tags to avoid being consumed by unrelated send/recv paths.
    """

    DEFAULT = 0
    HIRADIX_PP_SYNC = int.from_bytes(b"PpHi", byteorder="big")
    HIRADIX_PP_SYNC_HEADER = int.from_bytes(b"PpUH", byteorder="big")
    HIRADIX_PP_SYNC_PREFETCH = int.from_bytes(b"PpUP", byteorder="big")
    HIRADIX_PP_SYNC_QSIZES = int.from_bytes(b"PpUQ", byteorder="big")
    HIRADIX_PP_SYNC_READY = int.from_bytes(b"PpUR", byteorder="big")
    HIRADIX_PP_SYNC_VERIFY = int.from_bytes(b"PpUV", byteorder="big")
    HIRADIX_PP_SYNC_WRITE = int.from_bytes(b"PpUW", byteorder="big")
    HIRADIX_PP_SYNC_LOAD = int.from_bytes(b"PpUL", byteorder="big")
    HIRADIX_PP_SYNC_FINISH = int.from_bytes(b"PpUF", byteorder="big")
    HIRADIX_PP_COMMIT_LENGTH = int.from_bytes(b"PcLn", byteorder="big")
    HIRADIX_PP_COMMIT_FRAME = int.from_bytes(b"PcFm", byteorder="big")
    GRAMMAR_PP_SYNC = int.from_bytes(b"PpGr", byteorder="big")
