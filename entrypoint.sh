#!/bin/bash

source activate rapids && jupyter lab \
        --ip 0.0.0.0                               `# Run on localhost` \
        --allow-root                               `# Enable the use of sudo commands in the notebook` \
        --no-browser                               `# Do not launch a browser by default` \
        --NotebookApp.token=""                     `# Do not require token to access notebook` \
        --NotebookApp.password=""                  `# Do not require password to run jupyter server`
