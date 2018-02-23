# python-data-analysis

Python scripts for data analysis, mostly work in progress.

## Prerequisite

Python 3

## Dependencies

Install dependencies:

    pip3 install -r requirements.txt

## Run

Change into the `nwbib` directory:

    cd nwbib

Load sample NWBib data from the Lobid API:

    python3 nwbib_subjects_load.py

Run classification experiment:

    python3 nwbib_subjects_process.py

Run bulk classification (first run takes some time):

    python3 nwbib_subjects_bulk.py

Run a pipeline with cross-validation and hyperparameter optimization:

    python3 nwbib_subjects_pipeline.py

## License

[Eclipse Public License 2.0](http://www.eclipse.org/legal/epl-v20.html)