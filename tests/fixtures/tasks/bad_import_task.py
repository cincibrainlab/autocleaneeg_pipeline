from autoclean.core.task import Task
import non_existent_library

class BadImportTask(Task):
    """This task has a bad import."""
    def run(self):
        pass
