"""Safe domain errors; never include SQL, credentials, or record contents."""


class StorageError(Exception):
    pass


class DatabaseNotInitialized(StorageError):
    pass


class RecordNotFound(StorageError):
    pass


class VersionConflict(StorageError):
    pass


class SeedConflict(StorageError):
    pass
