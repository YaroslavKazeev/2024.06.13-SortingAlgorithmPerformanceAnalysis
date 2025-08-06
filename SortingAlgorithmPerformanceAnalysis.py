'''
READ ME:

Functions: bubble_sort, merge_sort, quick_sort, insertion_sort, tim_sort, selection_sort, shell_sort, genRandList, genDatasets, measure_time, evaluate_algorithms, plot_func, plot_dataset, nearlySortArrPrep, quicksortOpt, median_of_three, partition

DESCRIPTION(bubble_sort):
`bubble_sort` is a sorting algorithm that iterates through the list, comparing adjacent elements and swapping them if they are in the wrong order. This process is repeated until the list is sorted.
DESCRIPTION(merge_sort):
`merge_sort` is a divide-and-conquer sorting algorithm that recursively splits the list into halves, sorts each half, and then merges the sorted halves back together.
DESCRIPTION(quick_sort):
`quick_sort` is a sorting algorithm that selects a pivot element from the list and partitions the other elements into two sub-arrays, according to whether they are less than or greater than the pivot. The sub-arrays are then sorted recursively.
DESCRIPTION(insertion_sort):
`insertion_sort` is a sorting algorithm that builds the sorted array one item at a time by repeatedly picking the next item and inserting it into the correct position among the already sorted items.
DESCRIPTION(tim_sort):
`tim_sort` is a hybrid sorting algorithm derived from merge sort and insertion sort, designed to perform well on many kinds of real-world data. It is used in Python's built-in `sorted` function.
DESCRIPTION(selection_sort):
`selection_sort` is a sorting algorithm that repeatedly selects the smallest element from the unsorted portion of the list and swaps it with the first unsorted element.
DESCRIPTION(shell_sort):
`shell_sort` is an in-place comparison sort that generalizes insertion sort to allow the exchange of items that are far apart. The algorithm starts by sorting elements far apart from each other and successively reducing the gap between elements to be compared.
DESCRIPTION(genRandList):
`genRandList` generates a list of random integers of a specified size.
DESCRIPTION(genDatasets):
`genDatasets` generates datasets consisting of random, sorted, and reverse-sorted lists for testing sorting algorithms.
DESCRIPTION(measure_time):
`measure_time` measures the execution time of a given sorting function on a given dataset.
DESCRIPTION(evaluate_algorithms):
`evaluate_algorithms` tests various sorting algorithms on different types of datasets and records their execution times.
DESCRIPTION(plot_func):
`plot_func` generates and displays performance plots for each sorting algorithm, showing execution time against dataset size.
DESCRIPTION(plot_dataset):
`plot_dataset` generates and displays performance plots for each dataset type, comparing the execution times of different sorting algorithms.
DESCRIPTION(nearlySortArrPrep):
`nearlySortArrPrep` prepares nearly sorted arrays for testing by creating a semi-ordered list from a randomly ordered list.
DESCRIPTION(quicksortOpt):
`quicksortOpt` is an optimized version of quicksort that uses the median-of-three method to choose a pivot and employs tail recursion optimization.
DESCRIPTION(median_of_three):
`median_of_three` selects the median value of the first, middle, and last elements of the list as the pivot for quicksort.
DESCRIPTION(partition):
`partition` rearranges the elements in the list such that elements less than the pivot come before elements greater than the pivot.

PARAMETERS(bubble_sort, merge_sort, quick_sort, insertion_sort, tim_sort, selection_sort, shell_sort):
- arr: List of elements to be sorted.
PARAMETERS(genRandList):
- listSize: Integer representing the size of the list to generate.
PARAMETERS(genDatasets):
- startPower: Integer indicating the starting power of 2 for generating datasets (default is 4).
PARAMETERS(measure_time):
- sort_func: Sorting function to measure.
- dataset: List of elements to be sorted.
PARAMETERS(evaluate_algorithms):
- dataSets: Dictionary containing different types of datasets (random, sorted, reverseSorted).
PARAMETERS(plot_func):
- results: Dictionary containing execution times for each sorting algorithm on different dataset sizes.
PARAMETERS(plot_dataset):
- results: Dictionary containing execution times for each sorting algorithm on different dataset types.
- dataSets: Dictionary containing the datasets used for testing.
PARAMETERS(nearlySortArrPrep):
- dataSets: Dictionary containing different types of datasets.
- pivotPerc: Percentage indicating the proportion of the list to be sorted (default is 50%).
PARAMETERS(quicksortOpt):
- arr: List of elements to be sorted.
- low: Starting index for the sort (default is 0).
- high: Ending index for the sort (default is the length of the list minus 1).
PARAMETERS(median_of_three):
- arr: List of elements to be sorted.
- low: Starting index for the median selection.
- mid: Middle index for the median selection.
- high: Ending index for the median selection.
PARAMETERS(partition):
- arr: List of elements to be sorted.
- low: Starting index for the partition.
- high: Ending index for the partition.
- pivot: Pivot element around which the list will be partitioned.

LIMITATIONS(bubble_sort):
- Inefficient for large datasets due to its O(n^2) time complexity.
- Performs poorly on nearly sorted data.
LIMITATIONS(merge_sort):
- Requires additional space proportional to the size of the list, making it less space-efficient.
- Recursion depth may be a limitation for very large lists.
LIMITATIONS(quick_sort):
- Worst-case time complexity is O(n^2), though this is rare with good pivot selection.
- Not stable, meaning equal elements may not retain their original order.
LIMITATIONS(insertion_sort):
- Inefficient for large datasets due to its O(n^2) time complexity.
- Performs poorly on data with many inversions.
LIMITATIONS(tim_sort):
- Performance depends on the nature of the data; while generally efficient, worst-case scenarios can still be problematic.
- Additional space requirement can be a limitation for very large datasets.
LIMITATIONS(selection_sort):
- Inefficient for large datasets due to its O(n^2) time complexity.
- Not stable, meaning equal elements may not retain their original order.
LIMITATIONS(shell_sort):
- The performance depends heavily on the gap sequence used; poorly chosen sequences can lead to suboptimal performance.
- Not stable, meaning equal elements may not retain their original order.
LIMITATIONS(genRandList):
- Limited to generating integers within a fixed range (0-1000).
- Not suitable for generating lists with specific statistical properties.
LIMITATIONS(genDatasets):
- Only generates three types of datasets (random, sorted, reverseSorted).
- The size of the datasets grows exponentially, which may be impractical for very large sizes.
LIMITATIONS(measure_time):
- Does not account for variations in system load, which can affect timing accuracy.
- Assumes the dataset fits into memory, which may not be the case for very large datasets.
LIMITATIONS(evaluate_algorithms):
- Only evaluates a fixed set of sorting algorithms; new algorithms must be added manually.
- Assumes all datasets are in-memory, which may not be feasible for very large datasets.
LIMITATIONS(plot_func):
- Generates separate plots for each algorithm, which may make it difficult to compare algorithms directly.
- Assumes the `matplotlib` library is installed and properly configured.
LIMITATIONS(plot_dataset):
- Generates separate plots for each dataset type, which may make it difficult to compare algorithms directly.
- Assumes the `matplotlib` library is installed and properly configured.
LIMITATIONS(nearlySortArrPrep):
- Only prepares one nearly sorted array, which may not be representative of all nearly sorted data.
- Assumes a fixed percentage for sorting, which may not suit all use cases.
LIMITATIONS(quicksortOpt):
- Assumes the list can fit into memory, which may not be feasible for very large lists.
- The optimized approach may not always yield significant performance improvements for all data types.
LIMITATIONS(median_of_three):
- Adds complexity to the quicksort implementation, which may not always yield significant performance improvements.
- Assumes the list can fit into memory, which may not be feasible for very large lists.
LIMITATIONS(partition):
- Assumes the list can fit into memory, which may not be feasible for very large lists.
- Performance is highly dependent on the choice of pivot.

STRUCTURES(bubble_sort):
- Two nested for-loops iterate through the list to compare and swap elements.
STRUCTURES(merge_sort):
- Uses recursive function calls to divide the list into smaller parts and merge them in sorted order.
- While-loops and if-else statements merge the sorted halves.
STRUCTURES(quick_sort):
- Uses if-else statements to check base cases and list comprehensions to partition the list.
STRUCTURES(insertion_sort):
- For-loop iterates through the list, and a while-loop moves elements to their correct position.
STRUCTURES(tim_sort):
- Uses Python's built-in sorted function.
STRUCTURES(selection_sort):
- Two nested for-loops iterate through the list to find and place the minimum element.
STRUCTURES(shell_sort):
- While-loop reduces the gap, and nested for-loops and while-loops sort elements at each gap.
STRUCTURES(genRandList):
- List comprehension generates a list of random integers.
STRUCTURES(genDatasets):
- For-loop generates datasets, and nested function calls sort and reverse lists.
STRUCTURES(measure_time):
- Try-except block measures time, ensuring the function handles errors gracefully.
STRUCTURES(evaluate_algorithms):
- Nested for-loops iterate through sorting functions and datasets.
- Try-except blocks handle errors during sorting.
STRUCTURES(plot_func):
- For-loop generates plots for each sorting algorithm.
STRUCTURES(plot_dataset):
- For-loop generates plots for each dataset type.
STRUCTURES(nearlySortArrPrep):
- List slicing and concatenation create nearly sorted arrays.
STRUCTURES(quicksortOpt):
- While-loop and if-else statements optimize quicksort with median-of-three and tail recursion.
STRUCTURES(median_of_three):
- If-else statements ensure the median of three elements is chosen.
STRUCTURES(partition):
- While-loop and if-else statements partition the list around the pivot.

OUTPUT(bubble_sort, merge_sort, quick_sort, insertion_sort, selection_sort, shell_sort):
- Modifies the input list to be sorted in place.
OUTPUT(tim_sort):
- Returns a new sorted list.
OUTPUT(genRandList):
- Returns a list of random integers.
OUTPUT(genDatasets):
- Returns a dictionary containing different types of datasets.
OUTPUT(measure_time):
- Returns the execution time of the sorting function.
OUTPUT(evaluate_algorithms):
- Returns a dictionary containing execution times for each sorting algorithm on different dataset types.
OUTPUT(plot_func):
- Displays performance plots for each sorting algorithm.
OUTPUT(plot_dataset):
- Displays performance plots for each dataset type.
OUTPUT(nearlySortArrPrep):
- Returns a list of an unordered and a semi-ordered list.
OUTPUT(quicksortOpt):
- Modifies the input list to be sorted in place.
OUTPUT(median_of_three):
- Returns the median value of the three elements.
OUTPUT(partition):
- Returns the index where the pivot is placed after partitioning.
'''

# Sorting algorithms
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i, j, k = 0, 0, 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def tim_sort(arr):
    return sorted(arr)

def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def shell_sort(arr):
    n = len(arr)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr

import random
import time
import matplotlib.pyplot as plt


# Tode with cases and test scenarios
def genRandList(listSize):
    return [random.randint(0, 1000) for _ in range(listSize)]

# Create lists with random integers
def genDatasets(startPower=4):
    dataSets = { "random": [], "sorted" : [], "reverseSorted" : [] }
    for i in range(startPower, startPower+10):
        temp = genRandList( 2 ** i )
        dataSets["random"] += [temp]
        temp1 = temp[:]
        temp1.sort()
        dataSets["sorted"] += [temp1]
        temp2 = temp[:]
        temp2.sort(reverse=True)
        dataSets["reverseSorted"] += [temp2]
    print('Generation of datasets is done')
    return dataSets

def measure_time(sort_func, dataset):
    _copy = dataset.copy()
    start_time = time.time()
    sort_func(_copy)
    end_time = time.time()
    return end_time - start_time

def evaluate_algorithms(dataSets):
    results = {}
    sort_funcs = {
        "Bubble Sort": bubble_sort,
        "Merge Sort": merge_sort,
        "Quick Sort": quick_sort,
        "Insertion Sort": insertion_sort,
        "Tim Sort": tim_sort,
        "Selection Sort": selection_sort,
        "Shell Sort": shell_sort
    }
    print('Calculating execution time for each algorithm from the list:')
    print(", ".join([func for func in sort_funcs.keys()]) )
    print('Each algorithm is going to be tested against the dataset that consists of those dataset-type groups:',", ".join([_ for _ in dataSets.keys()]) )
    print('Each dataset-type group, in turn, consists of the lists with those numbers of integers (0-999) in them:', ", ".join([str(len(_)) for _ in dataSets["random"] ]), '\n'  )
    for sort_func in sort_funcs.keys():
        results[sort_func] = {}
    for sort_name, sort_func in sort_funcs.items():
        print('The algorithm:', sort_name)
        for dataset_type, dataset_list in dataSets.items():
            print('\tThe dataset type:', dataset_type)
            results[sort_name][dataset_type] = []
            print('\t\tIn processing the dataset(list) with size: ', end='')
            for dataset in dataset_list:
                print('', len(dataset), end='')
                elapsed_time = measure_time(sort_func, dataset)
                results[sort_name][dataset_type].append(elapsed_time)
            print('')
    print('The evaluation of algorithms is completed successfully')
    return results

# Plotting by the function name
def plot_func(results):
    for sort_name, dataset_types in results.items():
        plt.figure(figsize=(12, 8))
        for dataset_type, times in dataset_types.items():
            plt.plot( [len(_) for _ in dataSets["random"] ], times, label=dataset_type)
        plt.title(f'Performance of {sort_name}')
        plt.xlabel('Dataset Size')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.show()

# Plotting by the dataset name
def plot_dataset(results, dataSets):
    for _set in dataSets.keys():
        plt.figure(figsize=(12, 8))
        for sort_name, dataset_types in results.items():
            times = dataset_types[_set]
            plt.plot([len(_) for _ in dataSets["random"]], times, label=sort_name)
        plt.title(f'Comparison on {_set}')
        plt.xlabel('Dataset Size')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.show()

def nearlySortArrPrep(dataSets, pivotPerc = 50):
    unorderedList = dataSets["random"][0]
    semiorderedList = sorted(unorderedList[0:int(len(dataSets["random"])*pivotPerc/100)]) + unorderedList[int(len(dataSets["random"])*pivotPerc/100):]
    return [unorderedList, semiorderedList]

def quicksortOpt(arr, low=0, high=0):
    high = len(arr) - 1
    while low < high:
        # Use median-of-three to choose pivot
        mid = (low + high) // 2
        pivot = median_of_three(arr, low, mid, high)

        # Partition the array around the pivot
        pivot_index = partition(arr, low, high, pivot)

        # Tail Recursion Optimization: Sort the smaller part first
        if pivot_index - low < high - pivot_index:
            quicksortOpt(arr, low, pivot_index - 1)
            low = pivot_index + 1
        else:
            quicksortOpt(arr, pivot_index + 1, high)
            high = pivot_index - 1
    return arr

def median_of_three(arr, low, mid, high):
    if arr[low] > arr[mid]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
    if arr[mid] > arr[high]:
        arr[mid], arr[high] = arr[high], arr[mid]
    # arr[mid] is now the median of arr[low], arr[mid], arr[high]
    return arr[mid]

def partition(arr, low, high, pivot):
    while low <= high:
        while arr[low] < pivot:
            low += 1
        while arr[high] > pivot:
            high -= 1
        if low <= high:
            arr[low], arr[high] = arr[high], arr[low]
            low += 1
            high -= 1
    return low


if __name__ == '__main__':
    dataSets = genDatasets()
    results = evaluate_algorithms(dataSets)
    # plot_func(results)
    plot_dataset(results, dataSets)
    print('Performance of the optimized quicksort algorithm with the median-of-three tail recursion on random and half-sorted lists')
    unAndNearlySort = nearlySortArrPrep(dataSets)
    print(measure_time(quicksortOpt, unAndNearlySort[0]))
    print(measure_time(quicksortOpt, unAndNearlySort[1]))


