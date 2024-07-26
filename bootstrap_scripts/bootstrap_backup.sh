#!/bin/bash
# Define the current version
current_version="5.2.0"
# Path to the version.md file
version_file="version.md"
# Path to the notebooks directory
notebooks_dir="/srv/notebooks"
backup_dir="/srv/notebooks_backup_"$current_version
# Read the version from the version.md file
if [ ! -f "$version_file" ]; then
  echo "version.md file not found!"
  touch version.md
fi
file_version=$(cat $version_file)
# Compare versions
if [ "$file_version" \< "$current_version" ]; then
  echo "Version is less than $current_version. Creating backup and updating version."
  # Create a backup of the notebooks directory
  if [ -d "$notebooks_dir" ]; then
    cp -r $notebooks_dir $backup_dir
    echo "Backup created at $backup_dir."
    # Update the version number in the version.md file
    echo $current_version > $version_file
    echo "version.md updated to version $current_version."
  else
    echo "Notebooks directory not found!"
    exit 1
  fi
else
  echo "Version is up to date. No action taken."
fi