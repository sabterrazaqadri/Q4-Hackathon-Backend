# CLI Usage Examples for Retrieval & Pipeline Validation

This document provides examples of how to use the CLI for testing and validating the retrieval pipeline.

## Basic Query

To perform a basic query and retrieve relevant content chunks:

```bash
python -m src.cli.query_cli query "What is ROS 2?"
```

To specify the number of results to return:

```bash
python -m src.cli.query_cli query "Explain URDF" --top-k 3
```

To set a minimum score threshold:

```bash
python -m src.cli.query_cli query "AI robotics" --top-k 5 --threshold 0.7
```

## Pipeline Validation

To run basic pipeline validation with example queries:

```bash
python -m src.cli.query_cli validate-pipeline
```

## Comprehensive Validation Testing

To run comprehensive validation tests using predetermined test cases:

```bash
python -m src.cli.query_cli test-validation
```

To run validation with custom parameters:

```bash
python -m src.cli.query_cli test-validation --tolerance 0.1 --top-k 3 --verbose
```

- `--tolerance`: Acceptable difference threshold for validation comparisons (default: 0.05)
- `--top-k`: Number of top results to return for validation (default: 5)
- `--verbose`: Show detailed validation output (default: False)

## Example Output

When running a query:
```
Query: What is ROS 2?
Found 5 relevant chunks:
--------------------------------------------------
1. Score: 0.9456
   Content: ROS 2 (Robot Operating System 2) is a set of libraries and tools for...
   Source: https://example.com/ros2-intro
   Section: Introduction
   Module: Module 1
   Chapter: Chapter 1
--------------------------------------------------
```

When running validation:
```
Running comprehensive validation tests...
Running 3 test cases...

Testing: 'What is ROS 2 communication?'
  ✓ Valid (confidence: 1.00)

Testing: 'Explain URDF fundamentals'
  ✓ Valid (confidence: 0.95)

Testing: 'How to connect AI systems to robots?'
  ✓ Valid (confidence: 1.00)

Overall validation: PASSED
Average confidence: 0.98

🎉 All validation tests passed!
```