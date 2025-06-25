from autoclean.core.task import Task

class BadSyntaxTask(Task):
    """This task has a syntax error."""
    def run(self):
        pass

    def bad_method(self)  # Missing colon causes syntax error
        pass