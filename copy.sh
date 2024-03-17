#!/bin/bash

# Get the Desktop path
desktop_path="$HOME/Desktop"

# Search for files containing "knn" recursively
find . -name "*knn*" -exec cp {} "$desktop_path/" \;

echo "KNN files copied to Desktop (if any)."
