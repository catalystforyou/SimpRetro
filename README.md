# SimpRetro
SimpRetro is a simplified retrosynthesis planning tool which replaced the intricate single-step models with a straightforward template enumeration approach for retrosynthetic route planning on a real-world drug molecule dataset.

The foundation of this code is built upon Syntheseus (version 0.3.0), a retrosynthesis framework developed by Microsoft Research, which can be found at [Microsoft's Syntheseus repository](https://github.com/microsoft/syntheseus/).

## Environment Setup

To set up the required environment, please follow the instructions provided by Syntheseus. Specifically, for the results we have presented, you will need to set up the environments for both LocalRetro and RootAligned.

The single-step templates used in our approach are stored within `filtered_canonical_templates.json`. The test molecules, sourced from DrugHunter's Molecules of the Month, are available in `SMILES.txt`. Our in-stock molecules dataset can be downloaded [here](https://drive.google.com/file/d/1x33LmAizIdA5Dgw7IJp7_k7gRdNVv5cT/view?usp=sharing)

Additionally, please ensure the installation of the C++ version of RDChiral by running the following command: `conda install -c conda-forge -c ljn917 rdchiral_cpp`.

## Usage Instructions

To replicate the results presented in our paper, execute the script using `sh run_search.sh`.

If you wish to use SimpRetro for generating retrosynthetic routes for your own targets, ensure that your target molecules are saved in `targets.txt` and you wish to save the results in the `results/` directory. You can run the following command:

```bash
python syntheseus/cli/search.py inventory_smiles_file=targets.txt search_targets_file=SMILES.txt model_class=NoModel model_dir=filtered_canonical_templates.json time_limit_s=1800 search_algorithm=mcts results_dir=results/
```