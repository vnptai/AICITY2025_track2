# Domain-Aware Enhancements to Vision-Language Models for Urban Traffic Safety Question Answering

## Prepare
1. Install Package

```Shell
conda create -n cityllava python=3.10 -y
conda activate cityllava
cd AICITY2025_track2/
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install flash-attn --no-build-isolation
```

### Data Preparation

Firstly change the directory to `data_preprocess` and create the `data` directory.

```
cd data_preprocess
mkdir ./data
```

Please download the [wts-dataset](https://github.com/woven-visionai/wts-dataset). Then, put the datasets under `./data`. After unzip the datasets, the directory structure should be like this:

```
.
├── data
│   ├── BDD_PC_5k
│   │   ├── annotations
│   │   │   ├── bbox_annotated
│   │   │   ├── bbox_generated
│   │   │   └── caption
│   │   └── videos
│   ├── WTS
│   │   ├── annotations
│   │   │   ├── bbox_annotated
│   │   │   ├── bbox_generated
│   │   │   └── caption
│   │   └── videos
│   └── test_part
|       ├── view_used_as_main_reference_for_multiview_scenario.csv
│       ├── WTS_DATASET_PUBLIC_TEST
│       └── WTS_DATASET_PUBLIC_TEST_BBOX
└── ... # python and shell scripts
```

Then run the following script to process the test data:

```
bash prepare_data_test.sh
```
After this script is excuted, all the test data is prepared. You can download the fintuned model and run the inference step directly.

Run the following script to process the train data:

```
bash prepare_data_train.sh
```
<b>Note</b> that the Openai or Qwen API is required in "prepare_data_train.sh". You should modify the API_KEY in this script.

After the execution, the folder structure should be like this:

```
.
├── data
│   ├── BDD_PC_5k
│   │   ├── annotations
│   │   │   ├── bbox_annotated
│   │   │   ├── bbox_generated
│   │   │   └── caption
│   │   ├── bbox_global # BDD global views
│   │   │   ├── train
│   │   │   └── val
│   │   ├── bbox_local # BDD local views
│   │   │   ├── train
│   │   │   └── val
│   │   └── videos
│   ├── WTS
│   │   ├── annotations
│   │   │   ├── bbox_annotated
│   │   │   ├── bbox_generated
│   │   │   └── caption
│   │   ├── bbox_global # WTS global views
│   │   │   ├── train
│   │   │   └── val
│   │   ├── bbox_local # BDD local views
│   │   │   ├── train
│   │   │   └── val
│   │   └── videos
│   └── test_part
|       ├── view_used_as_main_reference_for_multiview_scenario.csv
│       ├── WTS_DATASET_PUBLIC_TEST
│       │   ├──bbox_global/test/public # WTS Test Images
│       │   ├──bbox_local/test/public
│       │   └──external/BDD_PC_5K
│       │       ├──bbox_global/test/public # BDD Test Images
│       │       └──bbox_local/test/public
│       └── WTS_DATASET_PUBLIC_TEST_BBOX
├── processed_anno
│   ├── frame_bbox_anno
│   │   ├── bdd_test_all_video_with_bbox_anno_first_frame.json
│   │   ├── bdd_train_all_video_with_bbox_anno_first_frame.json
│   │   ├── bdd_val_all_video_with_bbox_anno_first_frame.json
│   │   ├── wts_test_all_video_with_bbox_anno_first_frame.json
│   │   ├── wts_train_all_video_with_bbox_anno_first_frame.json
│   │   └── wts_val_all_video_with_bbox_anno_first_frame.json
│   ├── llava_format
│   │   ├── wts_bdd_train.json
│   │   └── wts_bdd_val.json
│   ├──best_view_for_test.json
│   └──perspective_test_images.json
└── test_processed_anno
│   ├── frame_bbox_anno
... # python and shell scripts
```

Then the processed annotations could be found under `./processed_anno`, and the train json is:

```
'./data/processed_anno/llava_format/wts_bdd_llava_qa_train_stage_filted_checked.json'
```


## Inference
We use caption model of [CityLLaVA](https://github.com/alibaba/AICITY2024_Track2_AliOpenTrek_CityLLaVA) to evaluate in WTS_TEST_SET

Firstly, the fine-tuned model could be download [here](https://modelscope.cn/models/AliOpenTrek/CityLLaVA). 

Secondly, you should check the parameters defined at `./scripts/inference.sh`, ensure that all essential files and model exist.

Now you can do inference on WTS_TEST_SET:

```
bash scripts/inference.sh
```

## Question-Answering

### Training
We use `ms-swift` to train Vision-Language Models. The processed data will be fed into framework then trained on labeled data. The example shown in `qa/examples/sft.py`

### Inference
After fine-tuning, we run inference on the test split using the QA inference pipeline.
The inference entry point is `qa/infer_qa.py`, which has been refactored to support argument parsing.

First, serving VLM checkpoints:
```
MODEL="OpenGVLab/InternVL3-78B" #CHECKPOINT_PATH

lmdeploy serve api_server $MODEL --chat-template internvl2_5 --server-port 23333 --tp 8
```

We provide a ready-to-use shell script:
```
bash qa/examples/infer_qa.py
```

