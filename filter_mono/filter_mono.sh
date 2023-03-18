#!/bin/sh

SRC_FILE_PATH=$1
TGT_FILE_PATH=$2

# check if two files are of the same length
SRC_N_LINES=`wc -l < $SRC_FILE_PATH`
TGT_N_LINES=`wc -l < $TGT_FILE_PATH`
if [ $SRC_N_LINES != $TGT_N_LINES ] ; then
    echo $SRC_FILE_PATH and $TGT_FILE_PATH has a different number of lines.
    exit 1
fi

# iterate and delete if there's a line that matches the filtering criteria
MIN_WORDS=2
MAX_WORDS=30
MAX_RATIO=1.5
MAX_RATIO_INV=$(bc <<< "scale=2;$MAX_RATIO^-1")

echo iterating through $SRC_FILE_PATH and $TGT_FILE_PATH ...
for (( i=$SRC_N_LINES; i>=1; i-- ))
do
    # number of lines
    SRC_N_WORDS=`sed -n $i\p $SRC_FILE_PATH | wc -w `
    TGT_N_WORDS=`sed -n $i\p $TGT_FILE_PATH | wc -w `
    RATIO=$(bc <<< "scale=1;$SRC_N_WORDS/$TGT_N_WORDS")
    if [ $SRC_N_WORDS -lt $MIN_WORDS -o $TGT_N_WORDS -lt $MIN_WORDS ] ; then
        sed -i '' $i\d $SRC_FILE_PATH
        sed -i '' $i\d $TGT_FILE_PATH
    elif [ $SRC_N_WORDS -gt $MAX_WORDS -o $TGT_N_WORDS -gt $MAX_WORDS ] ; then
        sed -i '' $i\d $SRC_FILE_PATH
        sed -i '' $i\d $TGT_FILE_PATH
    elif (( $(echo "$RATIO < $MAX_RATIO_INV" |bc -l) || $(echo "$MAX_RATIO < $RATIO" |bc -l) )) ; then
        sed -i '' $i\d $SRC_FILE_PATH
        sed -i '' $i\d $TGT_FILE_PATH
    fi
done

echo udah bang