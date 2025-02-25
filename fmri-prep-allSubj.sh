#!/bin/bash
# fmriprep-all-subjects.sh - Linux version for fMRIPrep BOLD image preprocessing

# Set strict error handling
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

bids_root_dir="/path/to/data" 
nthreads=4
mem=20 #20 gigabytes
fs_license="${bids_root_dir}/derivatives/license.txt"

# Docker paths
bids_root_docker="</path/to/data>"
fs_license_docker="/license.txt"
derivatives_dir="</path/to/data>/derivatives"

# Function to print section headers
print_header() {
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==========================================${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}$1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}$1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

# Check and pull the fMRIPrep Docker image if not available
check_fmriprep_image() {
    print_header "Checking fMRIPrep Docker image"
    
    if ! command -v docker >/dev/null 2>&1; then
        print_error "Docker is not installed. Please install Docker before running this script."
        exit 1
    fi
    
    if docker images | grep -q "nipreps/fmriprep"; then
        print_success "fMRIPrep Docker image is available."
    else
        print_warning "fMRIPrep Docker image not found. Pulling the latest version..."
        docker pull nipreps/fmriprep:latest
        if docker images | grep -q "nipreps/fmriprep"; then
            print_success "fMRIPrep Docker image pulled successfully."
        else
            print_error "Failed to pull fMRIPrep Docker image. Please check your internet connection."
            exit 1
        fi
    fi
}

# Processing a single subject
process_subject() {
    local subj=$1
    
    print_header "Processing subject: $subj"
    
    docker_cmd="docker run -it --rm \
        -v ${bids_root_dir}:${bids_root_docker} \
        -v ${fs_license}:${fs_license_docker} \
        --memory=${mem}g \
        nipreps/fmriprep:latest \
        ${bids_root_docker} ${bids_root_docker}/derivatives participant \
        --participant-label ${subj} \
        --skip-bids-validation \
        --md-only-boilerplate \
        --fs-license-file ${fs_license_docker} \
        --fs-no-reconall \
        --output-spaces MNI152NLin2009cAsym:res-2 \
        --nthreads ${nthreads} \
        --stop-on-first-crash"
    
    # Executing Docker command
    echo "Running: $docker_cmd"
    echo "Starting processing for subject $subj at $(date)"
    
    if eval $docker_cmd; then
        print_success "Completed processing for subject $subj at $(date)"
        return 0
    else
        print_error "Error processing subject $subj"
        read -p "Continue with next subject? (y/n) " choice
        if [[ "$choice" != "y" ]]; then
            print_error "Stopping processing due to error."
            exit 1
        fi
        return 1
    fi
}

# Main
print_header "fMRIPrep Processing for All Subjects"

# Check for fMRIPrep image and Docker installed
check_fmriprep_image

# Create derivatives directory if it doesn't exist
mkdir -p "${bids_root_dir}/derivatives"

# Find all subject directories
subject_dirs=($(find "$bids_root_dir" -maxdepth 1 -type d -name "sub-*" | sort))

# Filter out derivatives and code directories
filtered_subject_dirs=()
for dir in "${subject_dirs[@]}"; do
    base_dir=$(basename "$dir")
    if [[ "$base_dir" != "derivatives" && "$base_dir" != "code" ]]; then
        filtered_subject_dirs+=("$dir")
    fi
done

# Display found subjects
print_header "Subjects to Process"
for dir in "${filtered_subject_dirs[@]}"; do
    echo "  - $(basename "$dir")"
done
echo "Total subjects to process: ${#filtered_subject_dirs[@]}"

# Ask for confirmation
read -p "Do you want to proceed? (y/n) " confirmation
if [[ "$confirmation" != "y" ]]; then
    print_warning "Operation cancelled by user."
    exit 0
fi

# Initialize counters
total_subjects=${#filtered_subject_dirs[@]}
successful=0
failed=0

# Process each subject
for dir in "${filtered_subject_dirs[@]}"; do
    subj=$(basename "$dir" | sed 's/^sub-//') # Extract subject identification from folder name

    if process_subject "$subj"; then
        ((successful++))
    else
        ((failed++))
    fi
    
    # Logging progress
    current=$(($successful + $failed))
    echo "Progress: $current/$total_subjects subjects processed ($successful successful, $failed failed)"
done

# Final summary
print_header "Processing Summary"
echo "Total subjects: $total_subjects"
echo "Successfully processed: $successful"
echo "Failed: $failed"

if [ $failed -eq 0 ]; then
    print_success "All subjects processed successfully!"
else
    print_warning "$failed subjects failed processing. Check the logs for details."
fi

echo "Processing completed at $(date)"