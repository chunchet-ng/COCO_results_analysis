# COCO Results Analysis

This repo contains a simple toolkit to analyze object detection results of MS-COCO style JSONs.

## Description

Given COCO style groundtruth and prediction JSONs, you can analyze the per class Average Precision and Average Recall, and also plot the confusion matrix!

## Getting Started

### Dependencies

Install all the packages needed through pip:

```bash
pip install -r requirements.txt
```

### Executing program

Run the following command:
```python
python main.py --gt_path path_to_gt_json --pred_path path_to_pred_json --save_dir path_to_save_output
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

ConfusionMatrix class is adapted from:
* [object_detection_confusion_matrix](https://github.com/kaanakan/object_detection_confusion_matrix)

Per class Average Precision and Average Recall calculation are adapted from:
* [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
