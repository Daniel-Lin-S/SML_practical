"""Create configs with default parameters."""  

import os
import logging
import argparse

from evaluate_sml import get_top_dirs

parser = argparse.ArgumentParser(description="Create configs with default parameters.")

parser.add_argument(
    "--method",
    type=str,
    default="naive_bayes",
    help="The method to use, one of (naive_bayes, knn, lda, qda, xgboost_rf)",
)

def main():
    args = parser.parse_args()
    method_name = args.method
    top_dirs = get_top_dirs()
    
    for top_dir in top_dirs:
        
        path_to_output = os.path.join(top_dir, f"best_params_{method_name}.yaml")
        
        if os.path.exists(path_to_output):
            logging.info(f"File {path_to_output} already exists, skipping...")
            continue
        
        logging.info(f"Creating file {path_to_output}...")
        with open(path_to_output, "w") as file:
            file.write(f"{method_name}:\n")
            

if __name__ == "__main__":  
    main()