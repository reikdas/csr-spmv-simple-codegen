#!/bin/bash
set -e  # exit immediately if a command fails
set -u  # treat unset variables as errors

# List of URLs
urls=(
"https://suitesparse-collection-website.herokuapp.com/MM/Hollinger/mark3jac120sc.tar.gz"
"https://suitesparse-collection-website.herokuapp.com/MM/FIDAP/ex35.tar.gz"
"https://suitesparse-collection-website.herokuapp.com/MM/Shyy/shyy161.tar.gz"
"https://suitesparse-collection-website.herokuapp.com/MM/Hollinger/jan99jac120.tar.gz"
"https://suitesparse-collection-website.herokuapp.com/MM/ATandT/onetone2.tar.gz"
"https://suitesparse-collection-website.herokuapp.com/MM/Boeing/bcsstm39.tar.gz"
"https://suitesparse-collection-website.herokuapp.com/MM/Pothen/bodyy6.tar.gz"
"https://suitesparse-collection-website.herokuapp.com/MM/Grund/bayer10.tar.gz"
"https://suitesparse-collection-website.herokuapp.com/MM/Grund/poli_large.tar.gz"
)

cd matrices

for url in "${urls[@]}"; do
    echo "Processing $url ..."

    # Get the tarball name (e.g., ex35.tar.gz)
    tarball=$(basename "$url")

    # Download
    wget -q "$url" -O "$tarball"

    # Extract folder name (e.g., ex35)
    folder="${tarball%.tar.gz}"

    # Extract
    tar -xzf "$tarball"

    # Move the .mtx file to current directory
    if [ -f "$folder/$folder.mtx" ]; then
        mv "$folder/$folder.mtx" .
    else
        echo "Warning: $folder/$folder.mtx not found!"
    fi

    # Clean up
    rm -rf "$folder" "$tarball"

    echo "Done with $folder."
    echo
done

echo "All downloads complete!"
