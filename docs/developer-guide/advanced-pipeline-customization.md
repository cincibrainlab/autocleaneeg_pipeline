# Advanced Pipeline Customization

This guide explains advanced techniques for customizing and extending the AutoClean Pipeline architecture beyond the standard configuration options and step functions.

## Pipeline Extension Mechanisms

The AutoClean Pipeline provides several extension mechanisms:

1. **Custom Step Functions**: Create specialized processing operations
2. **Custom Tasks**: Define complete processing workflows
3. **Pipeline Subclassing**: Extend the core Pipeline class with new functionality
4. **Event Handlers**: Hook into pipeline events to add custom behavior
5. **Plugins**: Add entirely new functionality to the pipeline system

## Creating a Custom Pipeline Class

For the most extensive customization, you can create a subclass of the Pipeline class:

```python
from autoclean.core.pipeline import Pipeline
import mne
import numpy as np
import datetime

class MyCustomPipeline(Pipeline):
    """Custom pipeline with additional functionality."""
    
    def __init__(self, config=None, data_path=None, verbose=None, custom_param=None):
        # Call parent constructor first
        super().__init__(config=config, data_path=data_path, verbose=verbose)
        
        # Add custom attributes
        self.custom_param = custom_param
        self.custom_data = {}
        
        # Register custom event handlers
        self.on("file_loaded", self._on_file_loaded)
        self.on("processing_complete", self._on_processing_complete)
    
    def _on_file_loaded(self, file_path):
        """Custom handler for file_loaded event."""
        self.logger.info(f"Custom handler: File loaded from {file_path}")
        # Custom processing
    
    def _on_processing_complete(self, results):
        """Custom handler for processing_complete event."""
        self.logger.info(f"Custom handler: Processing completed with status: {results.get('success')}")
        # Custom reporting
    
    def custom_method(self, param1, param2=None):
        """Add a custom method to the pipeline.
        
        Parameters
        ----------
        param1 : type
            Description
        param2 : type, optional
            Description
        
        Returns
        -------
        result : type
            Description
        """
        # Implementation
        self.logger.info(f"Running custom method with param1={param1}, param2={param2}")
        # Custom processing
        result = param1 + (param2 or 0)
        
        # Update custom data
        self.custom_data["last_result"] = result
        self.custom_data["timestamp"] = str(datetime.datetime.now())
        
        return result
    
    def get_output_path(self, file_type, *args, **kwargs):
        """Override method to change output path behavior.
        
        Extends the parent method to handle custom file types.
        """
        if file_type == 'custom_report':
            # Custom logic for custom report file type
            base_dir = self.get_results_dir('reports')
            return base_dir / f"custom_report_{self.metadata.get('subject_id')}.pdf"
        
        # Fall back to parent implementation for other file types
        return super().get_output_path(file_type, *args, **kwargs)
    
    def process_file(self, file_path, task, **kwargs):
        """Override process_file to add custom behavior.
        
        Extends the parent method with pre and post processing.
        """
        # Pre-processing
        self.logger.info("Custom pre-processing in MyCustomPipeline")
        
        # Call parent implementation
        results = super().process_file(file_path, task, **kwargs)
        
        # Post-processing
        self.logger.info("Custom post-processing in MyCustomPipeline")
        self._generate_custom_report(results)
        
        return results
    
    def _generate_custom_report(self, results):
        """Generate a custom report.
        
        Private helper method for custom functionality.
        """
        # Implementation
        self.logger.info("Generating custom report")
        # Report generation logic
```

## Pipeline Events

The Pipeline class emits events at key points in processing. You can hook into these events to add custom behavior:

```python
from autoclean.core.pipeline import Pipeline

# Create a standard pipeline
pipeline = Pipeline()

# Define event handlers
def on_file_loaded(pipeline, file_path):
    """Handler for file_loaded event."""
    print(f"File loaded: {file_path}")
    print(f"Channels: {len(pipeline.raw.ch_names)}")
    # Additional processing

def on_ica_fitted(pipeline, ica):
    """Handler for ica_fitted event."""
    print(f"ICA fitted with {ica.n_components_} components")
    # Custom component visualization

def on_processing_error(pipeline, error, stage):
    """Handler for processing_error event."""
    print(f"Error in {stage}: {str(error)}")
    # Custom error handling or reporting

# Register event handlers
pipeline.on("file_loaded", on_file_loaded)
pipeline.on("ica_fitted", on_ica_fitted)
pipeline.on("processing_error", on_processing_error)

# Process file - events will be triggered
pipeline.process_file("subject01.raw", task="rest_eyesopen")
```

### Available Events

| Event | Description | Parameters |
|-------|-------------|------------|
| `initialization` | Pipeline initialized | pipeline |
| `config_loaded` | Configuration loaded | pipeline, config |
| `file_loaded` | Raw file loaded | pipeline, file_path |
| `preprocessing_start` | Preprocessing started | pipeline |
| `preprocessing_end` | Preprocessing completed | pipeline |
| `bad_channels_detected` | Bad channels detected | pipeline, bad_channels |
| `ica_fitted` | ICA fitted | pipeline, ica |
| `ica_components_excluded` | ICA components excluded | pipeline, excluded_components |
| `epochs_created` | Epochs created | pipeline, epochs |
| `evoked_created` | Evoked data created | pipeline, evoked |
| `processing_complete` | Processing completed | pipeline, results |
| `processing_error` | Error during processing | pipeline, error, stage |
| `state_saved` | Pipeline state saved | pipeline, file_path |
| `state_loaded` | Pipeline state loaded | pipeline, file_path |

## Custom Hooks

You can add custom hooks to enable specific behaviors at different stages of processing:

```python
from autoclean.core.pipeline import Pipeline
from autoclean.utils.hooks import register_hook, execute_hook

# Define a custom hook
@register_hook(name="post_filter_hook")
def my_custom_hook(pipeline, low_freq, high_freq):
    """Custom hook to execute after filtering.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    low_freq : float
        Lower frequency bound used for filtering
    high_freq : float
        Upper frequency bound used for filtering
    """
    pipeline.logger.info(f"Custom hook executed after filtering ({low_freq}-{high_freq}Hz)")
    
    # Custom processing or validation
    power = pipeline.raw.compute_psd().average()
    pipeline.metadata['post_filter_stats'] = {
        'mean_power': float(np.mean(power.get_data())),
        'peak_freq': float(power.freqs[np.argmax(power.get_data())])
    }

# Create a custom step function that uses the hook
def apply_filter_with_hook(pipeline, l_freq=1.0, h_freq=40.0):
    """Apply filter and execute the post-filter hook."""
    from autoclean.step_functions.preprocessing import apply_bandpass_filter
    
    # Apply the filter
    apply_bandpass_filter(pipeline, l_freq=l_freq, h_freq=h_freq)
    
    # Execute the hook
    execute_hook("post_filter_hook", pipeline, l_freq, h_freq)
```

## Pipeline Plugins

For more extensive customization, you can create pipeline plugins that add entirely new functionality:

```python
from autoclean.utils.plugin import PipelinePlugin

class MyPipelinePlugin(PipelinePlugin):
    """Custom plugin adding new functionality to the pipeline."""
    
    name = "my_plugin"
    
    def __init__(self, pipeline):
        """Initialize the plugin with the pipeline instance."""
        super().__init__(pipeline)
        
        # Add plugin-specific attributes
        self.plugin_data = {}
        
        # Register event handlers
        pipeline.on("file_loaded", self._on_file_loaded)
    
    def _on_file_loaded(self, pipeline, file_path):
        """Handle file_loaded event."""
        self.logger.info(f"Plugin: File loaded from {file_path}")
    
    def plugin_method(self, param1, param2=None):
        """Add a new method available through the plugin.
        
        Parameters
        ----------
        param1 : type
            Description
        param2 : type, optional
            Description
        
        Returns
        -------
        result : type
            Description
        """
        # Implementation
        pipeline = self.pipeline
        pipeline.logger.info(f"Running plugin method with param1={param1}")
        
        # Access pipeline data
        if pipeline.raw is not None:
            # Do something with raw data
            pass
        
        return result

# Register and use the plugin
from autoclean.core.pipeline import Pipeline
from autoclean.utils.plugin import register_plugin

# Register the plugin
register_plugin(MyPipelinePlugin)

# Create pipeline and access plugin
pipeline = Pipeline()
plugin = pipeline.plugins.my_plugin

# Use plugin functionality
result = plugin.plugin_method(42)
```

## Custom Processing Strategies

You can implement custom processing strategies by defining how pipeline operations are executed:

```python
from autoclean.core.pipeline import Pipeline
from autoclean.step_functions import preprocessing, artifacts, ica

class ChunkProcessingStrategy:
    """Strategy for processing data in chunks to reduce memory usage."""
    
    def __init__(self, chunk_size='60s', overlap='5s'):
        """Initialize with chunk size and overlap.
        
        Parameters
        ----------
        chunk_size : str
            Size of each chunk (e.g. '60s', '1m', '5000samples')
        overlap : str
            Overlap between chunks (e.g. '5s', '500samples')
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def process(self, pipeline, task):
        """Process the data in chunks.
        
        Parameters
        ----------
        pipeline : Pipeline
            The pipeline instance
        task : Task
            The task to execute
        
        Returns
        -------
        dict
            Processing results
        """
        # Validate inputs
        if pipeline.raw is None:
            raise ValueError("Raw data must be loaded first")
        
        # Calculate chunk size and overlap in samples
        sfreq = pipeline.raw.info['sfreq']
        chunk_size_samples, overlap_samples = self._parse_duration(
            pipeline.raw, self.chunk_size, self.overlap)
        
        # Get total duration
        n_samples = len(pipeline.raw.times)
        
        # Initialize results
        results = []
        
        # Process each chunk
        for start_idx in range(0, n_samples, chunk_size_samples - overlap_samples):
            end_idx = min(start_idx + chunk_size_samples, n_samples)
            
            # Skip last chunk if too small
            if end_idx - start_idx < chunk_size_samples / 2:
                break
            
            # Log progress
            pipeline.logger.info(f"Processing chunk {start_idx}-{end_idx}")
            pipeline.progress("chunking", 
                             percentage=100 * start_idx / n_samples,
                             message=f"Processing chunk {start_idx}-{end_idx}")
            
            # Create chunk
            chunk_raw = pipeline.raw.copy().crop(
                tmin=start_idx / sfreq,
                tmax=(end_idx - 1) / sfreq,
                include_tmax=True
            )
            
            # Create a temporary pipeline for the chunk
            chunk_pipeline = Pipeline(config=pipeline.config)
            chunk_pipeline.raw = chunk_raw
            
            # Process the chunk with the given task
            task.run(chunk_pipeline)
            
            # Store results
            results.append({
                'start_sample': start_idx,
                'end_sample': end_idx,
                'data': chunk_pipeline.raw.get_data(),
                'metadata': chunk_pipeline.metadata
            })
        
        # Combine chunk results
        pipeline.logger.info("Combining chunk results")
        combined_results = self._combine_results(pipeline, results)
        
        return combined_results
    
    def _parse_duration(self, raw, duration_str, overlap_str):
        """Parse duration strings to sample counts."""
        # Implementation
        # ...
        return chunk_size_samples, overlap_samples
    
    def _combine_results(self, pipeline, chunk_results):
        """Combine results from all chunks."""
        # Implementation
        # ...
        return combined_results

# Use the custom processing strategy
pipeline = Pipeline()
pipeline.load_raw("subject01.raw")

# Create the strategy
strategy = ChunkProcessingStrategy(chunk_size='60s', overlap='5s')

# Create a task
from autoclean.tasks.resting_eyes_open import RestingEyesOpen
task = RestingEyesOpen()

# Process with the strategy
results = strategy.process(pipeline, task)
```

## Memory-Optimized Processing

For large datasets, you can optimize memory usage:

```python
from autoclean.core.pipeline import Pipeline
import mne
import gc

class MemoryOptimizedPipeline(Pipeline):
    """Pipeline optimized for processing large datasets with limited memory."""
    
    def __init__(self, *args, **kwargs):
        # Get memory limit
        self.memory_limit_gb = kwargs.pop('memory_limit_gb', 4)
        super().__init__(*args, **kwargs)
    
    def process_file(self, file_path, task, **kwargs):
        """Process a file with memory optimization."""
        # Estimate file size before loading
        file_info = mne.io.read_raw_info(file_path)
        estimated_gb = (
            file_info['nchan'] * 
            file_info['highpass'] * 
            4  # Bytes per float32 value
        ) / (1024**3)  # Convert to GB
        
        self.logger.info(f"Estimated memory usage: {estimated_gb:.2f} GB")
        
        if estimated_gb > self.memory_limit_gb:
            # Use chunked processing
            self.logger.info(f"File exceeds memory limit of {self.memory_limit_gb} GB, using chunked processing")
            return self._process_in_chunks(file_path, task, **kwargs)
        else:
            # Use standard processing
            self.logger.info("Using standard processing")
            return super().process_file(file_path, task, **kwargs)
    
    def _process_in_chunks(self, file_path, task, **kwargs):
        """Process a file in chunks to reduce memory usage."""
        # Implementation
        # ...
        
        # Force garbage collection between operations
        gc.collect()
        
        return results
```

## Pipeline Middleware

You can add middleware layers that modify the behavior of the pipeline without changing its core:

```python
from autoclean.core.pipeline import Pipeline
from functools import wraps
import time

class PipelineMiddleware:
    """Base class for pipeline middleware."""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._wrap_methods()
    
    def _wrap_methods(self):
        """Wrap pipeline methods with middleware."""
        pass
    
    def before_method(self, method_name, *args, **kwargs):
        """Called before a method is executed."""
        pass
    
    def after_method(self, method_name, result, *args, **kwargs):
        """Called after a method is executed."""
        pass
    
    def on_error(self, method_name, error, *args, **kwargs):
        """Called when a method raises an exception."""
        pass

class LoggingMiddleware(PipelineMiddleware):
    """Middleware for detailed logging of pipeline operations."""
    
    def _wrap_methods(self):
        """Wrap pipeline methods for logging."""
        methods_to_wrap = [
            'process_file', 'load_raw', 'save_raw', 'fit_ica',
            'apply_ica', 'create_epochs', 'create_evoked'
        ]
        
        for method_name in methods_to_wrap:
            if hasattr(self.pipeline, method_name):
                original_method = getattr(self.pipeline, method_name)
                
                @wraps(original_method)
                def wrapped_method(original=original_method, name=method_name, *args, **kwargs):
                    self.before_method(name, *args, **kwargs)
                    start_time = time.time()
                    
                    try:
                        result = original(*args, **kwargs)
                        elapsed = time.time() - start_time
                        self.after_method(name, result, elapsed, *args, **kwargs)
                        return result
                    except Exception as e:
                        self.on_error(name, e, *args, **kwargs)
                        raise
                
                setattr(self.pipeline, method_name, wrapped_method)
    
    def before_method(self, method_name, *args, **kwargs):
        """Log before method execution."""
        self.pipeline.logger.info(f"Starting {method_name} with args: {args}, kwargs: {kwargs}")
    
    def after_method(self, method_name, result, elapsed, *args, **kwargs):
        """Log after method execution."""
        self.pipeline.logger.info(f"Completed {method_name} in {elapsed:.2f}s")
    
    def on_error(self, method_name, error, *args, **kwargs):
        """Log method errors."""
        self.pipeline.logger.error(f"Error in {method_name}: {str(error)}")

# Apply middleware to a pipeline
pipeline = Pipeline()
middleware = LoggingMiddleware(pipeline)

# Now all pipeline methods will be logged in detail
pipeline.process_file("subject01.raw", task="rest_eyesopen")
```

## Multi-Task Processing

For complex processing workflows that combine multiple tasks, you can create a composite task:

```python
from autoclean.core.task import Task
from autoclean.core.pipeline import Pipeline

class CompositeTask(Task):
    """A task that combines multiple subtasks."""
    
    def __init__(self, tasks, name="composite"):
        """Initialize with a list of tasks.
        
        Parameters
        ----------
        tasks : list of Task
            Subtasks to run in sequence
        name : str, optional
            Name for this composite task
        """
        super().__init__(name=name)
        self.tasks = tasks
    
    def run(self, pipeline):
        """Run all subtasks in sequence.
        
        Parameters
        ----------
        pipeline : Pipeline
            The pipeline instance
        """
        # Log the start of the composite task
        pipeline.logger.info(f"Starting composite task with {len(self.tasks)} subtasks")
        
        # Run each subtask
        subtask_results = []
        for i, task in enumerate(self.tasks):
            pipeline.logger.info(f"Running subtask {i+1}/{len(self.tasks)}: {task.name}")
            pipeline.progress("composite_task", 
                             percentage=100 * i / len(self.tasks),
                             message=f"Running subtask {task.name}")
            
            # Run the subtask
            result = task.run(pipeline)
            subtask_results.append(result)
        
        # Log completion
        pipeline.logger.info(f"Completed composite task with {len(self.tasks)} subtasks")
        
        # Update metadata
        pipeline.metadata['composite_task'] = {
            'num_subtasks': len(self.tasks),
            'subtask_names': [task.name for task in self.tasks],
            'results': subtask_results
        }
        
        return pipeline

# Use the composite task
from autoclean.tasks.resting_eyes_open import RestingEyesOpen
from autoclean.tasks.assr_default import AssrDefault

# Create subtasks
task1 = RestingEyesOpen()
task2 = AssrDefault()

# Create composite task
composite = CompositeTask([task1, task2], name="rest_and_assr")

# Run with the pipeline
pipeline = Pipeline()
pipeline.process_file("subject01.raw", task=composite)
```

## Conclusion

By leveraging these advanced customization techniques, you can extend the AutoClean Pipeline to meet your specific research needs. Whether you need custom processing algorithms, memory optimization, or entirely new functionality, the pipeline architecture provides flexible extension points.

For specific examples of these customization techniques, see the [Examples](../examples) directory, which contains working code for various extension scenarios.

Related Documentation:
- [Creating Custom Step Functions](creating-custom-step-functions.md)
- [Pipeline API Reference](../api-reference/pipeline.md)
- [Task API Reference](../api-reference/task.md)
- [Step Functions API Reference](../api-reference/step-functions.md) 