from autoclean import Pipeline
from autoclean.utils.logging import configure_logger, logger
from pathlib import Path
import time
from datetime import datetime

def print_header(text):
    logger.log("HEADER", "=" * 80)
    logger.log("HEADER", text.center(80))
    logger.log("HEADER", "=" * 80)

def run_test(file_path, task_name):
    logger.info(f"\n📁 Testing file: {file_path.name}")
    logger.info(f"🔧 Task: {task_name}")
    
    start_time = time.time()
    try:
        pipeline.process_file(
            file_path=file_path,
            task=task_name
        )
        duration = time.time() - start_time
        logger.success(f"✅ Test passed in {duration:.2f} seconds")
        return True
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Test failed after {duration:.2f} seconds")
        logger.error(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    # Setup
    OUTPUT_DIR = Path("C:/Users/Gam9LG/Documents/Autoclean_dev")
    CONFIG_PATH = Path("C:/Users/Gam9LG/Documents/Autoclean_dev/testing_config.yaml")

    # Configure logging
    configure_logger(verbose="INFO", output_dir=OUTPUT_DIR)
    
    print_header(f"AutoClean Pipeline Tests - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize pipeline
    pipeline = Pipeline(
        autoclean_dir=OUTPUT_DIR,
        autoclean_config=CONFIG_PATH,
        verbose="HEADER"  # Reduce pipeline verbosity since we have our own logging
    )

    # Define test cases
    test_cases = [
        (Path('C:/Users/Gam9LG/Documents/Autoclean_dev/testing_data/resting_eyes_open.set'), "RestingEyesOpen"),
        (Path('C:/Users/Gam9LG/Documents/Autoclean_dev/testing_data/resting_eyes_open.raw'), "RestingEyesOpen"),
        (Path('C:/Users/Gam9LG/Documents/Autoclean_dev/testing_data/hbcd_mmn.set'), "HBCD_MMN"),
        (Path('C:/Users/Gam9LG/Documents/Autoclean_dev/testing_data/hbcd_mmn.mff'), "HBCD_MMN"),
        (Path('C:/Users/Gam9LG/Documents/Autoclean_dev/testing_data/mouse_assr.set'), "MouseXdatAssr"),
    ]

    # Run tests
    results = []
    for file_path, task_name in test_cases:
        success = run_test(file_path, task_name)
        results.append((file_path.name, task_name, success))

    # Print summary
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
















