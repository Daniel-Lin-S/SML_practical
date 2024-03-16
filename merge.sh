#!/bin/bash

# Paths to directories A and B
dirA="new_configs/sml_configs"
dirB="configs/sml_configs"
mergedDir="temp_configs/sml_configs"

# Remove the Merged directory if it exists, then create a fresh one
rm -rf "$mergedDir"
mkdir -p "$mergedDir"

# Copy everything from A to Merged
cp -R "$dirA/"* "$mergedDir"

# Copy contents of B, merging with Merged
cp -Rn "$dirB/"* "$mergedDir"

echo "Directories A and B have been merged into $mergedDir"
