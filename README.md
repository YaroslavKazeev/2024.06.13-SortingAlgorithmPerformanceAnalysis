# Sorting Algorithm Performance Analysis

This project provides a comprehensive analysis of various sorting algorithms' performance across different types of datasets. It includes implementations of popular sorting algorithms and tools to measure and visualize their performance characteristics.

## Quick Start

```bash
# Install dependencies
pipenv install matplotlib
```

The project requires the dependency **matplotlib** for performance visualization and plotting

## Features

- **Multiple Sorting Algorithms**: Bubble Sort, Merge Sort, Quick Sort, Insertion Sort, Tim Sort, Selection Sort, Shell Sort
- **Optimized Quick Sort**: Includes an optimized version with median-of-three pivot selection and tail recursion optimization
- **Dataset Generation**: Creates random, sorted, and reverse-sorted datasets of varying sizes
- **Performance Measurement**: Measures execution time for each algorithm on different dataset types
- **Visualization**: Generates performance plots comparing algorithms and dataset types
- **Nearly Sorted Data Testing**: Special functionality for testing algorithms on partially sorted data

### Dependencies

- **matplotlib**: For performance visualization
- **random**: Built-in Python module for dataset generation
- **time**: Built-in Python module for performance measurement

### Program Output

1. Generate datasets of different sizes (2^4 to 2^13 elements)
2. Test all sorting algorithms on random, sorted, and reverse-sorted data
3. Display performance plots comparing algorithms
4. Test the optimized quicksort on random and nearly-sorted data

## Algorithms Implemented

### Basic Sorting Algorithms

1. **Bubble Sort** - O(n²) time complexity, simple but inefficient for large datasets
2. **Merge Sort** - O(n log n) time complexity, stable and predictable performance
3. **Quick Sort** - O(n log n) average case, O(n²) worst case, in-place sorting
4. **Insertion Sort** - O(n²) time complexity, efficient for small datasets
5. **Selection Sort** - O(n²) time complexity, simple in-place sorting
6. **Shell Sort** - O(n log n) to O(n²) depending on gap sequence
7. **Tim Sort** - Python's built-in sorting algorithm, hybrid of merge and insertion sort

### Optimized Quick Sort

The project includes an optimized version of quicksort with:

- **Median-of-three pivot selection**: Reduces worst-case scenarios
- **Tail recursion optimization**: Improves stack usage for large datasets

## Dataset Types

The analysis tests algorithms on three types of datasets:

1. **Random Data**: Unordered lists of random integers (0-999)
2. **Sorted Data**: Pre-sorted lists in ascending order
3. **Reverse Sorted Data**: Pre-sorted lists in descending order
4. **Nearly Sorted Data**: Partially sorted lists for specialized testing
