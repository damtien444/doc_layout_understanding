from data_loader_coco_image import DocumentLayoutAnalysisDataset


pipeline_predictions = DocumentLayoutAnalysisDataset(
    root_dir="/home/tiendq/Desktop/DocRec/2_data_preparation/4_test_data/images",
    annotation_file="/home/tiendq/PycharmProjects/DeepLearningDocReconstruction/1_data_preparation/annotated_input_coco.json"
)

print(pipeline_predictions[0])


