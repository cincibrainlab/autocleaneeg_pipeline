# test_imports.py
try:
    import autoclean
    print("Successfully imported autoclean")
    print(f"Version: {autoclean.__version__}")
    
    from autoclean.core import pipeline, task
    print("Successfully imported core modules")
    
    from autoclean.step_functions import io, reports
    print("Successfully imported step_functions modules")
    
    from autoclean.mixins import signal_processing
    print("Successfully imported mixins modules")
    
    from autoclean.utils import config, logging
    print("Successfully imported utils modules")
    
    print("All imports successful")
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()