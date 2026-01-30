"""
This script is designed to compute departmental statistics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import sys


###########################################################################################################
# ARGOS functions
###########################################################################################################

def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.
    Args:
        file_path (Path): Path to the CSV file.
    Returns:
        DataFrame: Loaded DataFrame or None if file does not exist.
    """
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return None
    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
        return None

def load_and_concatenate_data(terms, base_path_template):
    """
    Load and concatenate data from multiple files, excluding empty DataFrames.
    Args:
        terms (list): List of terms.
        base_path_template (str): Template for the file path.
    Returns:
        DataFrame: Combined DataFrame or an empty DataFrame if no valid data found.
    """
    dataframes = []
    for term in terms:
        file_path = Path(base_path_template.format(term=term))
        df = load_csv(file_path)
        print(f'Processing File: {file_path}')
        if df is not None and not df.empty:
            dataframes.append(df)
    if dataframes:
        concatenated_df = pd.concat(dataframes, ignore_index=True)
        print(f"Number of rows in concatenated_df is {len(concatenated_df)}")
        return concatenated_df
    else:
        print("No valid data files found.")
        return pd.DataFrame()


# Define the function for questions
def combine_stats(group, metrics):
    """
    Combine metrics (e.g., counts, averages, and standard deviations) across groups.
    Args:
        group (DataFrame): Grouped DataFrame.
        metrics (list): List of metrics to process. Each metric should be a dict containing:
            - 'name': Base name of the metric (e.g., 'Q3', 'GPA').
            - 'count_col': Column name for counts associated with the metric.
            - 'avg_col': Column name for averages associated with the metric.
            - 'stddev_col': Column name for standard deviations associated with the metric.
    Returns:
        Series: Combined results for the metrics.
    """
    results = {}
    for metric in metrics:
        count_col = metric['count_col']
        avg_col = metric['avg_col']
        stddev_col = metric['stddev_col']

        total_count = group[count_col].sum()
        if total_count == 0:
            # Handle zero total count
            results[f"{metric['name']}_count"] = 0
            results[f"{metric['name']}_avg"] = np.nan
            results[f"{metric['name']}_stddev"] = np.nan
        else:
            # Weighted average
            weighted_avg = (group[count_col] * group[avg_col]).sum() / total_count
            # Combined standard deviation
            combined_variance = (
                (group[count_col] * (group[stddev_col]**2 + (group[avg_col] - weighted_avg)**2)).sum()
                / total_count
            )
            combined_stddev = np.sqrt(combined_variance)
            # Store results
            results[metric['count_col']] = total_count
            results[metric['avg_col']] = weighted_avg
            results[metric['stddev_col']] = combined_stddev
    return pd.Series(results)

def compute_quantiles(data):
    """
    Compute the 25%, 50%, and 75% quantile points for the data.
    Args:
        data (array-like): Input data.
    Returns:
        dict: Dictionary with quantile values.
    """
    quantiles = {
        '25%': np.percentile(data, 25),
        '50%': np.percentile(data, 50),
        '75%': np.percentile(data, 75)
    }
    return quantiles

def compute_full_quantiles_dataframe(data_df, metrics):
    """
    Compute quantiles for all departments, levels, and specified metrics, returning as a DataFrame.
    Args:
        data_df (DataFrame): Combined DataFrame with all data.
        metrics (list): List of metrics to compute quantiles for.
    Returns:
        DataFrame: DataFrame with quantiles for each department, level, and metric.
    """
    quantile_records = []
    dept_list = data_df['Dept'].unique()  # Get unique departments from the DataFrame
    for dept in dept_list:
        for level in data_df['Level'].unique():
            filtered_df = data_df.loc[(data_df['Dept'] == dept) & (data_df['Level'] == level)]
            if not filtered_df.empty:
                record = {'Dept': dept, 'Level': level}
                for metric in metrics:
                    n_samples = filtered_df[f'{metric}_count'].to_numpy(dtype=np.float64)
                    q_samples = filtered_df[f'{metric}_avg'].to_numpy(dtype=np.float64)

                    # Expand q_samples by n_samples
                    expanded_data = np.repeat(q_samples, n_samples.astype(int))

                    # Compute quantiles
                    quantiles = compute_quantiles(expanded_data)
                    record[f'{metric}_q25'] = quantiles['25%']
                    record[f'{metric}_q50'] = quantiles['50%']
                    record[f'{metric}_q75'] = quantiles['75%']

                quantile_records.append(record)

    return pd.DataFrame(quantile_records)

###########################################################################################################

Terms = [
    # '202031', '202111', '202121',
    # '202131', '202211', '202221',
    '202231', '202311', # '202321',
    '202331', '202411', # '202421',
    '202431', '202511'
]

# Questions to process
question_num = [3, 4, 5, 7, 8, 9]

# Define metrics for processing
question_metrics = [
    {'name': f"Q{q}", 'count_col': f"Q{q}_count", 'avg_col': f"Q{q}_avg", 'stddev_col': f"Q{q}_stddev"}
    for q in question_num
]
general_metrics = [
    {'name': 'GPA', 'count_col': 'Count', 'avg_col': 'GPA', 'stddev_col': 'STD'}
]
all_metrics = question_metrics + general_metrics


# List desired CSV files in the directory
college_id = 'EN'  # Engineering College ID
base_path = 'DATA_Evaluations'
output_path = 'DATA_DeptStats'

# Ensure output directory exists
Path(output_path).mkdir(parents=True, exist_ok=True)

# Load ASSESSMENT Data
assessment_df = load_and_concatenate_data(Terms, base_path + f'/Data-{college_id}-{{term}}.csv')
print(f'ASSESSMENT: {len(assessment_df)}')

if not assessment_df.empty:
    # Apply the combined function iteratively
    grouped = assessment_df.groupby(['Dept', 'Level'])
    results = []
    for (dept, level), group in grouped:
        # Apply combine_stats to each group
        group_result = combine_stats(group.drop(columns=['Dept', 'Level']), metrics=all_metrics)
        # Add the group key back to the result
        group_result['Dept'] = dept
        group_result['Level'] = level
        results.append(group_result)

    # Combine all results into a DataFrame
    combined_stats = pd.DataFrame(results)
    print(f'Number of rows in combined_stats is {len(combined_stats)}')

    # Save combined statistics to a CSV file
    stats_output_file = f'{output_path}/{college_id}-departmental-statistics.csv'
    print(f"Saving File: {stats_output_file}")
    try:
        combined_stats.to_csv(stats_output_file, index=False)
    except PermissionError:
        print(f"Error: Permission denied writing to '{stats_output_file}'.")
    except Exception as e:
        print(f"Error writing '{stats_output_file}': {e}")

    # Compute quantiles for all departments, levels, and metrics
    metrics = ['Q3', 'Q4', 'Q5', 'Q7', 'Q8', 'Q9', 'GPA']
    assessment_df.rename(columns={'GPA': 'GPA_avg'}, inplace=True)
    assessment_df.rename(columns={'Enrollment': 'GPA_count'}, inplace=True)

    quantile_input = assessment_df.rename(
        columns={'GPA': 'GPA_avg', 'Enrollment': 'GPA_count'}
    ).copy()
    quantiles_df = compute_full_quantiles_dataframe(quantile_input, metrics)

    quantiles_output_file = f'{output_path}/{college_id}-quantiles.csv'
    print(f"Saving File: {quantiles_output_file}")
    try:
        quantiles_df.to_csv(quantiles_output_file, index=False)
    except PermissionError:
        print(f"Error: Permission denied writing to '{quantiles_output_file}'.")
    except Exception as e:
        print(f"Error writing '{quantiles_output_file}': {e}")
else:
    print("No valid data to process.")
