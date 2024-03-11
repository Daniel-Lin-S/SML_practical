#!/bin/bash

# the line above is a shebang line, which tells the system to run the script using bash


# Install dependencies
pip install --user -r requirements.txt
pip install xgboost

# Base directory for output
base_dir="configs/sml_configs"

# Ensure the base directory exists
mkdir -p "$base_dir"

# Loop through reduce_method options
for reduce_method in lda none; do
    # Define n_components array based on the reduce_method
    if [ "$reduce_method" = "pca" ]; then
        declare -a n_components=(5 10 15 20)
    elif [ "$reduce_method" = "lda" ]; then
        declare -a n_components=(7)
    else
        # For 'none', we don't need to specify n_components, but we'll run it once for consistency
        declare -a n_components=(none)
    fi
    
    # Loop through n_components
    for n in "${n_components[@]}"; do
        # Define output directory based on parameters
        if [ "$reduce_method" = "none" ]; then
            output_dir="${base_dir}/${reduce_method}"
        else
            output_dir="${base_dir}/${reduce_method}_${n}"
        fi

        # Ensure the output directory exists
        mkdir -p "$output_dir"

        # Run the script with the current set of parameters
        if [ "$reduce_method" = "none" ]; then
            echo "Running: python run_sml.py --reduce_method $reduce_method --output_dir $output_dir"
            python run_sml.py --reduce_method "$reduce_method" --output_dir "$output_dir"
        else
            echo "Running: python run_sml.py --reduce_method $reduce_method --n_components $n --output_dir $output_dir"
            python run_sml.py --reduce_method "$reduce_method" --n_components "$n" --output_dir "$output_dir"
        fi
    done
done

echo "All tasks completed."









