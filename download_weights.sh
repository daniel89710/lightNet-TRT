#!/bin/bash

set -e

filelist=(\
"https://drive.google.com/file/d/1qTBQ0BkIYqcyu1BwC54_Z9T1_b702HKf/view" \
"https://drive.google.com/file/d/1ttdVtlDiPun13EQCB4Nyls3Q8w5aXg1i/view" \
"https://drive.google.com/file/d/1OGZApPeNH7K08-89eJ8tGhzA9kkRFLii/view" \
"https://drive.google.com/file/d/1dytYnqS4h_5YK73tr6DOZTUYUEKZYsFe/view" \
)

filenames=(\
"lightNet-BDD100K-1280x960.weights" \
"lightNet-BDD100K-det-semaseg-1280x960.weights" \
"lightNet-BDD100K-1280x960-chPruning.weights" \
"lightNet-BDD100K-chPruning-det-semaseg-1280x960.weights"\
)

# Download weights
for ((i=0;i<${#filelist[@]};++i)); do
    fileid="$(echo ${filelist[i]} | cut -d'/' -f6)"
    filename=${filenames[i]}
    echo "Downloading ${filename}..."
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ./configs/${filename}
done

# Remove cookie
rm cookie