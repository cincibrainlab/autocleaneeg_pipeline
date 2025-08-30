import tempfile
import time
from datetime import datetime
from pathlib import Path

from autoclean import Pipeline
from autoclean.utils.logging import configure_logger, logger


def print_header(text):
    logger.log("HEADER", "=" * 80)
    logger.log("HEADER", text.center(80))
    logger.log("HEADER", "=" * 80)


def create_test_task_files(temp_dir):
    """Create test task files for the new Python task system."""
    temp_dir = Path(temp_dir)

    # Simple test task for basic functionality
    simple_task_content = """
from autoclean.core.task import Task
from typing import Any, Dict

# Embedded configuration
config = {
    'resample_step': {'enabled': True, 'value': 250},
    'filtering': {'enabled': True, 'value': {'l_freq': 1, 'h_freq': 40}},
    'reference_step': {'enabled': True, 'value': 'average'},
    'epoch_settings': {'enabled': True, 'value': {'tmin': -1, 'tmax': 1}}
}

class SimpleTestTask(Task):
    def __init__(self, config: Dict[str, Any]):
        self.settings = globals()['config']
        super().__init__(config)
    
    def run(self):
        self.import_raw()
        self.run_basic_steps(export=True)
        self.create_regular_epochs(export=True)
"""

    # Advanced test task for export functionality
    advanced_task_content = """
from autoclean.core.task import Task
from typing import Any, Dict

# Advanced configuration with ICA
config = {
    'resample_step': {'enabled': True, 'value': 250},
    'filtering': {'enabled': True, 'value': {'l_freq': 1, 'h_freq': 40, 'notch_freqs': [60]}},
    'reference_step': {'enabled': True, 'value': 'average'},
    'ICA': {'enabled': True, 'value': {'method': 'infomax', 'n_components': 10}},
    'epoch_settings': {'enabled': True, 'value': {'tmin': -1, 'tmax': 1}}
}

class AdvancedTestTask(Task):
    def __init__(self, config: Dict[str, Any]):
        self.settings = globals()['config']
        super().__init__(config)
    
    def run(self):
        self.import_raw()
        self.run_basic_steps(export=True)
        self.run_ica(export=True)
        self.create_regular_epochs(export=True)
        self.gfp_clean_epochs(export=True)
"""

    # HBCD MMN task for specific format testing
    hbcd_task_content = """
from autoclean.core.task import Task
from typing import Any, Dict

# HBCD-specific configuration
config = {
    'resample_step': {'enabled': True, 'value': 250},
    'filtering': {'enabled': True, 'value': {'l_freq': 0.5, 'h_freq': 40}},
    'reference_step': {'enabled': False, 'value': 'average'},
    'epoch_settings': {'enabled': True, 'value': {'tmin': -0.2, 'tmax': 0.8}}
}

class HBCDTestTask(Task):
    def __init__(self, config: Dict[str, Any]):
        self.settings = globals()['config']
        super().__init__(config)
    
    def run(self):
        self.import_raw()
        self.run_basic_steps(export=True)
        self.create_eventid_epochs()  # Event-based epochs for MMN
"""

    # Write task files
    simple_task_path = temp_dir / "simple_test_task.py"
    advanced_task_path = temp_dir / "advanced_test_task.py"
    hbcd_task_path = temp_dir / "hbcd_test_task.py"

    simple_task_path.write_text(simple_task_content)
    advanced_task_path.write_text(advanced_task_content)
    hbcd_task_path.write_text(hbcd_task_content)

    return {
        "simple": simple_task_path,
        "advanced": advanced_task_path,
        "hbcd": hbcd_task_path,
    }


def run_yaml_test(pipeline, file_path, task_name):
    """Run test with traditional YAML-based pipeline."""
    logger.info(f"\n📁 [YAML] Testing file: {file_path.name}")
    logger.info(f"🔧 [YAML] Task: {task_name}")

    start_time = time.time()
    try:
        pipeline.process_file(file_path=file_path, task=task_name)
        duration = time.time() - start_time
        logger.success(f"✅ [YAML] Test passed in {duration:.2f} seconds")
        return True
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ [YAML] Test failed after {duration:.2f} seconds")
        logger.error(f"Error: {str(e)}")
        return False


def run_python_test(pipeline, file_path, task_file_path, task_class_name):
    """Run test with new Python task file system."""
    logger.info(f"\n📁 [PYTHON] Testing file: {file_path.name}")
    logger.info(f"🔧 [PYTHON] Task file: {task_file_path.name}")
    logger.info(f"🏷️  [PYTHON] Task class: {task_class_name}")

    start_time = time.time()
    try:
        # Register the Python task file
        task_name = pipeline.add_task(str(task_file_path))
        logger.info(f"📝 [PYTHON] Registered task: {task_name}")

        # Process the file
        pipeline.process_file(file_path=file_path, task=task_class_name)
        duration = time.time() - start_time
        logger.success(f"✅ [PYTHON] Test passed in {duration:.2f} seconds")
        return True
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ [PYTHON] Test failed after {duration:.2f} seconds")
        logger.error(f"Error: {str(e)}")
        return False


def test_pipeline_features(pipeline):
    """Test new pipeline features."""
    logger.info("\n🧪 Testing pipeline features...")

    tests_passed = 0
    total_tests = 0

    # Test 1: List tasks
    total_tests += 1
    try:
        tasks = pipeline.list_tasks()
        logger.info(f"📋 Available tasks: {tasks}")
        tests_passed += 1
        logger.success("✅ list_tasks() works correctly")
    except Exception as e:
        logger.error(f"❌ list_tasks() failed: {e}")

    # Test 2: List stage files
    total_tests += 1
    try:
        stages = pipeline.list_stage_files()
        logger.info(f"📂 Available stages: {stages}")
        tests_passed += 1
        logger.success("✅ list_stage_files() works correctly")
    except Exception as e:
        logger.error(f"❌ list_stage_files() failed: {e}")

    return tests_passed, total_tests


def run_test(file_path, task_name):
    """Run both YAML and Python tests for a given file and task."""
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 Testing: {file_path.name} with {task_name}")
    logger.info(f"{'='*60}")

    # Check if file exists
    if not file_path.exists():
        logger.warning(f"⚠️  File not found: {file_path}")
        return False

    # Initialize pipelines
    yaml_success = False
    python_success = False

    # Test 1: YAML-based approach (traditional)
    try:
        yaml_pipeline = Pipeline(
            autoclean_dir=OUTPUT_DIR / "yaml_tests",
            autoclean_config=CONFIG_PATH,
            verbose="ERROR",  # Reduce verbosity for tests
        )
        yaml_success = run_yaml_test(yaml_pipeline, file_path, task_name)
    except Exception as e:
        logger.error(f"❌ YAML pipeline setup failed: {e}")

    # Test 2: Python task file approach (new)
    try:
        # Create Python-only pipeline
        python_pipeline = Pipeline(
            autoclean_dir=OUTPUT_DIR / "python_tests", verbose="ERROR"
        )

        # Create temporary Python task file for this test
        with tempfile.TemporaryDirectory() as temp_dir:
            task_files = create_test_task_files(temp_dir)

            # Use simple task for testing
            simple_task_file = Path(temp_dir) / "simple_test_task.py"
            python_success = run_python_test(
                python_pipeline, file_path, simple_task_file, "SimpleTestTask"
            )
    except Exception as e:
        logger.error(f"❌ Python pipeline setup failed: {e}")

    # Overall result
    overall_success = yaml_success or python_success  # Pass if either works

    if yaml_success and python_success:
        logger.success("🎉 Both YAML and Python approaches succeeded!")
    elif yaml_success:
        logger.success("✅ YAML approach succeeded")
        logger.warning("⚠️  Python approach had issues")
    elif python_success:
        logger.success("✅ Python approach succeeded")
        logger.warning("⚠️  YAML approach had issues")
    else:
        logger.error("❌ Both approaches failed")

    return overall_success


if __name__ == "__main__":
    # Setup
    OUTPUT_DIR = Path("C:/Users/Gam9LG/Documents/Autoclean_testing")
    CONFIG_PATH = Path(
        "C:/Users/Gam9LG/Documents/Autoclean_testing/autoclean_config.yaml"
    )

    # Configure logging
    configure_logger(verbose="INFO", output_dir=OUTPUT_DIR)

    print_header(
        f"AutoClean Pipeline Tests - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Initialize pipeline
    pipeline = Pipeline(
        autoclean_dir=OUTPUT_DIR,
        autoclean_config=CONFIG_PATH,
        verbose="HEADER",  # Reduce pipeline verbosity since we have our own logging
    )

    # Define test cases
    test_cases = [
        (
            Path(
                "C:/Users/Gam9LG/Documents/Autoclean_testing/testing_data/resting_eyes_open.set"
            ),
            "RestingEyesOpen",
        ),
        (
            Path(
                "C:/Users/Gam9LG/Documents/Autoclean_testing/testing_data/resting_eyes_open.raw"
            ),
            "RestingEyesOpen",
        ),
        (
            Path(
                "C:/Users/Gam9LG/Documents/Autoclean_testing/testing_data/hbcd_mmn.set"
            ),
            "HBCD_MMN",
        ),
        (
            Path(
                "C:/Users/Gam9LG/Documents/Autoclean_testing/testing_data/hbcd_mmn.mff"
            ),
            "HBCD_MMN",
        ),
        (
            Path(
                "C:/Users/Gam9LG/Documents/Autoclean_testing/testing_data/mouse_assr.set"
            ),
            "MouseXdatAssr",
        ),
    ]

    # Run tests
    results = []
    for file_path, task_name in test_cases:
        success = run_test(file_path, task_name)
        results.append((file_path.name, task_name, success))

    # Print summary
    configure_logger(verbose="INFO", output_dir=OUTPUT_DIR)
    print_header("Test Summary")
    total_tests = len(results)
    passed_tests = sum(1 for _, _, success in results if success)

    logger.info(f"\nTotal tests: {total_tests}")
    logger.success(f"Passed: {passed_tests}")
    if passed_tests < total_tests:
        logger.error(f"Failed: {total_tests - passed_tests}")

    if passed_tests == total_tests:
        logger.success("\n🎉 All tests passed successfully!")
    else:
        logger.error("\n❌ Some tests failed. Check the logs for details.")
