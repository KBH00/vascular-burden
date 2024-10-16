
INPUT_DIR="/mnt/d/VascularData/data/ADNI" #original data (e.g., dcm)
OUTPUT_DIR="/mnt/d/VascularData/data/nii" #converted data path (e.g., nii or nii.gz)

process_nifti() {
	local input_file="$1"
	local base_name=$(basename "$input_file" .nii)
	local input_dir=$(dirname "$input_file")  

			local output_subdir="$input_dir"
			echo "Processing in directory: $output_subdir"

				    # Apply BET for skull stripping
				    local stripped_output="$output_subdir/${base_name}.nii.gz"
				    echo "Stripped output file: $stripped_output"

				    if [ -f "$input_file" ]; then
					    bet "$input_file" "$stripped_output" -f 0.5
				    else
					    echo "Input file $input_file not found."
					    return
				    fi
				    if [ ! -f "$stripped_output" ]; then
					    echo "BET failed for $input_file"
					    return
				    fi

				    local binary_mask_output="$output_subdir/${base_name}_binary_mask.nii.gz"
				    fslmaths "$stripped_output" -thr 0.001 -bin "$binary_mask_output"
				    if [ ! -f "$binary_mask_output" ]; then
					    echo "Binary mask creation failed for $input_file"
					    return
				    fi

				    local cleaned_output="$output_subdir/${base_name}_cleaned.nii.gz"
				    fslmaths "$stripped_output" -mas "$binary_mask_output" "$cleaned_output"
				    echo "$cleand_output //// Done ! ! !"
				    #rm -f "$stripped_output" "$binary_mask_output"
			    }

				export -f process_nifti

				find "$INPUT_DIR" -type d | while read SUBDIR; do
				if ls "$SUBDIR"/*.dcm 1> /dev/null 2>&1; then
					RELATIVE_PATH="${SUBDIR#$INPUT_DIR/}"
					mkdir -p "$OUTPUT_DIR/$RELATIVE_PATH"
					dcm2niix -o "$OUTPUT_DIR/$RELATIVE_PATH" "$SUBDIR"
					find "$OUTPUT_DIR/$RELATIVE_PATH" -type f \( -name "*.nii" \) -exec bash -c 'process_nifti "$1"' _ {} \;
				fi
			done

			echo "Processing completed. All output files are saved in $OUTPUT_DIR."
