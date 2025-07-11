#!/bin/bash

# Check if a filename is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename>"
    echo "Example: $0 input.txt"
    exit 1
fi

input_file="$1"

# Check if the file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' not found."
    exit 1
fi

# Create output filename
output_file="${input_file%.*}_with_commas.${input_file##*.}"

# Add comma only to lines that contain just '}'
sed '/^}$/s/$/,/' "$input_file" > "$output_file"

echo "Commas added to lines containing only '}'."
echo "Original file: $input_file"
echo "Modified file: $output_file"
