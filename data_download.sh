FILE="data_download.txt"
DEST="data_files"

if [ ! -d "$DEST" ]; then 
    echo "Creating Directory $DEST"
    mkdir -p "$DEST"
fi

while read -r line || [ -n "$line" ]; do
    echo "Downloading $line"

    filename="${line#*private/}"
    if [ "$filename" = "$line" ]; then
        filename="${line#*public/}"
    fi
    filename="${filename%%\?*}"

    echo "Saving as: $filename"

    curl -L -o "$DEST/$filename" "$line"

done < "$FILE"

echo "Downloaded all files"