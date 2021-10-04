#!/bin/bash
#    ___           __________________  ___________
#   / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
#  / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/
# /_/ \___/_/   /___/\___/___/___/_/|_/\___/___/
#
# Author : Benjamin Blundell - k1803390@kcl.ac.uk
#
# Simple script to upload our results to the tensorboard
#
# https://misc.flogisoft.com/bash/tip_colors_and_formatting

RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Colour

echo -e "\U1F6A7 " ${YELLOW}GENERATING STATS${NC} "\U1F6A7"

if [ $# < 1 ]
then
  echo "Please pass in the directory of results"
  exit 1
fi

#VARS=`getopt -o ni:g: --long normalise,input:,ground: -- "$@"`
#eval set -- "$vars"

base=$1
echo $base
subdir="${1##*/}"

# Finally, perform the upload to the benjamin.computer server
echo -e "\U1F680 " ${GREEN}"UPLOADING TO" $subdir${NC} "\U1F680"
rsync -raz --update --progress $base --exclude "model.pth" oni@benjamin.computer:/srv/http/ai.benjamin.computer/public/experiments/.
#ssh oni@benjamin.computer "sudo chown -R www-data:oni /srv/http/ai.benjamin.computer/public/experiments/$subdir"
echo -e "\U1F37B"
